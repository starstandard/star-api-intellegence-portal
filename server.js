import 'dotenv/config';
import express from 'express';
import OpenAI from 'openai';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const MCP_SERVER_URL = process.env.MCP_SERVER_URL;
const MODEL = process.env.OPENAI_MODEL || 'gpt-5';

if (!OPENAI_API_KEY) console.error('Missing OPENAI_API_KEY');
if (!MCP_SERVER_URL) console.error('Missing MCP_SERVER_URL');

const client = new OpenAI({ apiKey: OPENAI_API_KEY });
const app = express();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

/* --------------------------------------------------
 * TIMEOUTS
 * -------------------------------------------------- */

async function withTimeout(promise, ms, label = 'Operation') {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });

  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

/* --------------------------------------------------
 * MCP
 * -------------------------------------------------- */

let rpcId = 1;

function parseMcpResponse(text, contentType = '') {
  const trimmed = String(text || '').trim();
  if (!trimmed) throw new Error('Empty MCP response');

  if (contentType.includes('application/json') && trimmed.startsWith('{')) {
    return JSON.parse(trimmed);
  }

  if (trimmed.startsWith('{')) {
    return JSON.parse(trimmed);
  }

  const dataLines = trimmed
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.replace(/^data:\s*/, '').trim())
    .filter(Boolean);

  if (!dataLines.length) {
    throw new Error(`Invalid MCP response: ${trimmed}`);
  }

  return JSON.parse(dataLines[dataLines.length - 1]);
}

async function callMcp(method, params = {}) {
  if (!MCP_SERVER_URL) {
    throw new Error('Missing MCP_SERVER_URL');
  }

  const response = await withTimeout(
    fetch(MCP_SERVER_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream'
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: rpcId++,
        method,
        params
      })
    }),
    20000,
    `MCP ${method} request`
  );

  const contentType = response.headers.get('content-type') || '';
  const text = await withTimeout(
    response.text(),
    10000,
    `MCP ${method} body read`
  );

  if (!response.ok) {
    throw new Error(`MCP ${method} failed: ${text}`);
  }

  const parsed = parseMcpResponse(text, contentType);

  if (parsed.error) {
    throw new Error(`MCP ${method} error: ${JSON.stringify(parsed.error)}`);
  }

  return parsed.result;
}

async function tool(name, args = {}) {
  return callMcp('tools/call', {
    name,
    arguments: args
  });
}

/* --------------------------------------------------
 * CACHE
 * -------------------------------------------------- */

const cache = new Map();
const CACHE_MS = 5 * 60 * 1000;

function makeCacheKey(name, args = {}) {
  return `${name}:${JSON.stringify(args)}`;
}

async function toolCached(name, args = {}) {
  const key = makeCacheKey(name, args);
  const now = Date.now();
  const cached = cache.get(key);

  if (cached && now - cached.ts < CACHE_MS) {
    return cached.value;
  }

  const value = await tool(name, args);
  cache.set(key, { value, ts: now });
  return value;
}

/* --------------------------------------------------
 * HELPERS
 * -------------------------------------------------- */

function detectDomain(message = '') {
  const m = String(message).toLowerCase();

  if (
    m.includes('multi-point-inspection') ||
    m.includes('multi point inspection') ||
    m.includes('inspection') ||
    m.includes('mpi')
  ) {
    return 'multi-point-inspection';
  }

  if (m.includes('appointment')) {
    return 'appointment';
  }

  return null;
}

function wantsCapabilities(message = '') {
  const m = String(message).toLowerCase();
  return (
    m.includes('capabilities') ||
    m.includes('business capabilities') ||
    m.includes('service advisor') ||
    m.includes('service lane') ||
    m.includes('advisor-facing') ||
    m.includes('daily workflow')
  );
}

function wantsOverview(message = '') {
  const m = String(message).toLowerCase();
  return (
    m.includes('overview') ||
    m.includes('what is') ||
    m.includes('summarize') ||
    m.includes('summary') ||
    m.includes('tell me about') ||
    m.includes('executive overview')
  );
}

function wantsSchemas(message = '') {
  return String(message).toLowerCase().includes('schema');
}

function wantsEndpoints(message = '') {
  const m = String(message).toLowerCase();
  return (
    m.includes('endpoint') ||
    m.includes('endpoints') ||
    m.includes('operation') ||
    m.includes('operations') ||
    m.includes('url') ||
    m.includes('path')
  );
}

function safeParseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function extractOperations(result) {
  return result?.operations || result?.structuredContent?.operations || [];
}

function extractSchemas(result) {
  return result?.schemas || result?.structuredContent?.schemas || [];
}

function extractSections(answer) {
  const text = String(answer || '').trim();
  if (!text) return [];

  const lines = text.split('\n');
  const sections = [];
  let current = null;

  const isHeader = (line) =>
    /^\s*\d+\.\s+/.test(line) ||
    /^\s*[A-Z][A-Za-z0-9 /&()-]{2,}:\s*$/.test(line);

  for (const raw of lines) {
    const line = raw.trim();

    if (!line) {
      if (current) current.body.push('');
      continue;
    }

    if (isHeader(line)) {
      if (current) sections.push(current);
      current = { title: line.replace(/:\s*$/, ''), body: [] };
    } else {
      if (!current) current = { title: 'Overview', body: [] };
      current.body.push(line);
    }
  }

  if (current) sections.push(current);

  return sections
    .map((s) => ({ title: s.title, body: s.body.join('\n').trim() }))
    .filter((s) => s.title || s.body);
}

/* --------------------------------------------------
 * BUILT-IN DOMAIN INTELLIGENCE
 * -------------------------------------------------- */

const DOMAIN_CONFIG = {
  appointment: {
    overview_blurb:
      'The Appointment API supports dealership service scheduling, intake coordination, and service-lane preparation.',
    rich_overview: `1. What it is
The Appointment API represents the dealership service scheduling and intake layer. It helps coordinate customer bookings, requested services, timing expectations, and operational readiness before the vehicle enters the service lane.

2. Core concepts
The API centers on the service appointment as the planning object that connects customer intent, vehicle context, dealership availability, and requested work. It provides the structure needed to align the front-end scheduling experience with dealership operations.

3. Main resources
Typical resources include the appointment itself, requested services, customer and vehicle references, timing details, and supporting status information needed by dealership systems.

4. Typical workflow
A customer books service, the dealership validates the request, the advisor or system confirms timing and service intent, and the service lane uses the appointment context to prepare intake and execution.

5. What to explore next
Look at business capabilities, representative endpoints, and schemas related to scheduling, intake, and service preparation.`,
    capability_cards: [
      {
        title: 'Schedule appointment',
        description: 'Create and manage service appointments for customers and vehicles.',
        business_guidance:
          'This capability covers the initial booking process. For a service advisor or dealership workflow owner, it represents the front door into service operations. It ensures that customer requests are captured with enough structure to support downstream preparation and execution.',
        advisor_actions: [
          'Review requested services and timing',
          'Confirm customer and vehicle context',
          'Make sure appointment data supports intake readiness'
        ],
        customer_impact:
          'Customers get a clearer booking experience and more accurate service expectations.',
        workflow_stage: 'Scheduling',
        prompt:
          'Explain how to schedule and manage appointments in the Appointment API for a dealership service advisor'
      },
      {
        title: 'Confirm service needs',
        description: 'Validate requested services, timing, and customer expectations.',
        business_guidance:
          'This capability helps the dealership turn a raw booking into an operationally useful service request. It is where the advisor validates what the customer wants, checks whether the request is complete, and aligns the appointment with dealership readiness.',
        advisor_actions: [
          'Validate requested work',
          'Check timing and operational fit',
          'Reduce ambiguity before service intake'
        ],
        customer_impact:
          'Customers receive more accurate service planning and fewer surprises at intake.',
        workflow_stage: 'Pre-intake validation',
        prompt:
          'Explain how a service advisor confirms service needs using the Appointment API'
      },
      {
        title: 'Prepare service lane',
        description: 'Use appointment context to prepare the dealership service workflow.',
        business_guidance:
          'This capability connects the appointment to the dealership operating model. It helps the service lane prepare capacity, people, and expectations before the customer arrives.',
        advisor_actions: [
          'Review upcoming appointments',
          'Use appointment data to prepare intake',
          'Coordinate service-lane readiness'
        ],
        customer_impact:
          'Customers experience smoother intake and better operational coordination.',
        workflow_stage: 'Service preparation',
        prompt:
          'Explain how the Appointment API helps prepare the service lane'
      },
      {
        title: 'Coordinate customer communication',
        description: 'Support reminders, confirmations, and status coordination.',
        business_guidance:
          'This capability supports the communication layer around appointments. It helps the dealership align customer expectations before arrival and reduce missed or unclear service interactions.',
        advisor_actions: [
          'Confirm appointment context',
          'Support reminders and updates',
          'Align service expectations'
        ],
        customer_impact:
          'Customers receive clearer communication and better appointment confidence.',
        workflow_stage: 'Customer coordination',
        prompt:
          'Explain how customer communication is supported by the Appointment API'
      }
    ],
    workflow_map: [
      {
        step: 'Schedule appointment',
        detail: 'Customer booking and advisor intake',
        prompt: 'Explain the scheduling flow of the Appointment API'
      },
      {
        step: 'Confirm service needs',
        detail: 'Advisor validates request and timing',
        prompt: 'Explain advisor validation workflow in the Appointment API'
      },
      {
        step: 'Prepare service lane',
        detail: 'Appointment data supports dealer operations',
        prompt: 'Explain how the Appointment API supports service-lane preparation'
      }
    ]
  },

  'multi-point-inspection': {
    overview_blurb:
      'The Multi-Point Inspection API supports the vehicle inspection lifecycle, from starting inspections through findings, recommendations, customer decisions, and completion.',
    rich_overview: `1. What it is
The Multi-Point Inspection API is a dealership service workflow domain that supports structured vehicle health checks. It covers the lifecycle from launching an inspection through technician findings, recommendations, customer decisions, and final completion.

2. Core concepts
The domain revolves around the inspection itself, what is being inspected, what was found, and what action should be recommended or approved. For dealership teams, it acts as the business structure that turns technician observations into advisor-facing guidance and customer decisions.

3. Main resources
Typical resources include the inspection record, findings, condition evidence, media, recommendations, approval outcomes, and workflow status. These objects help connect technical inspection work to customer-facing service decisions.

4. Typical workflow
An inspection is started from service intake or repair-order context. The technician records findings and supporting evidence. The advisor reviews the results, prepares recommendations, communicates with the customer, captures approvals or declines, and moves approved work into execution.

5. What to explore next
Review business capabilities, workflow stages, representative endpoints, and schemas related to findings, recommendations, approvals, and inspection completion.`,
    capability_cards: [
      {
        title: 'Start inspection workflow',
        description: 'Begin the MPI process from intake, appointment context, or repair-order workflow.',
        business_guidance:
          'This capability launches the inspection lifecycle. In dealership terms, it is the point where the vehicle moves from intake context into structured inspection activity. It gives technicians and advisors a common workflow anchor for inspection progress and results.',
        advisor_actions: [
          'Initiate inspection from service context',
          'Ensure the inspection is tied to the right vehicle and visit',
          'Use the inspection as the operational anchor for downstream findings'
        ],
        customer_impact:
          'Customers benefit from a structured and traceable inspection process.',
        workflow_stage: 'Inspection initiation',
        prompt:
          'Explain how to start the inspection workflow in the multi-point-inspection API for a dealership service advisor'
      },
      {
        title: 'Track inspection progress',
        description: 'Monitor inspection state, technician progress, and workflow status.',
        business_guidance:
          'This capability gives advisors visibility into where the inspection stands. It matters operationally because advisors need to know when findings are available, whether the inspection is still in progress, and when they can begin customer-facing recommendation work.',
        advisor_actions: [
          'Monitor inspection lifecycle status',
          'Coordinate timing between technician work and advisor communication',
          'Use progress visibility to manage service-lane flow'
        ],
        customer_impact:
          'Customers receive more timely updates and better expectation management.',
        workflow_stage: 'Inspection monitoring',
        prompt:
          'Explain how to track inspection progress in the multi-point-inspection API for a dealership service advisor'
      },
      {
        title: 'Review findings and media',
        description: 'Review technician findings, notes, photos, and condition evidence.',
        business_guidance:
          'This capability is where raw technician observations become advisor-usable business information. Advisors need more than a status flag. They need findings, evidence, and context they can understand and communicate clearly.',
        advisor_actions: [
          'Review condition findings',
          'Use media and notes as supporting evidence',
          'Translate technical observations into customer-facing explanation'
        ],
        customer_impact:
          'Customers receive clearer, evidence-based explanations of vehicle condition.',
        workflow_stage: 'Findings review',
        prompt:
          'Explain how to review findings and media in the multi-point-inspection API for a dealership service advisor'
      },
      {
        title: 'Prepare recommendations',
        description: 'Turn findings into advisor-ready repair recommendations and next steps.',
        business_guidance:
          'This capability converts inspection results into action. It is where the advisor interprets findings, organizes recommended work, and prepares a customer-ready proposal tied to vehicle condition and service priorities.',
        advisor_actions: [
          'Turn findings into recommended work',
          'Organize repair guidance by business relevance',
          'Prepare customer-facing next steps'
        ],
        customer_impact:
          'Customers receive clearer recommendations and understand why work is being proposed.',
        workflow_stage: 'Recommendation building',
        prompt:
          'Explain how to prepare recommendations in the multi-point-inspection API for a dealership service advisor'
      },
      {
        title: 'Capture customer approval',
        description: 'Record customer decisions, approvals, and communication outcomes.',
        business_guidance:
          'This capability captures the business outcome of the advisor-customer interaction. It is critical because inspection value is realized only when customer decisions are recorded clearly enough to drive execution, deferral, or closure.',
        advisor_actions: [
          'Record approvals and declines',
          'Track customer communication outcomes',
          'Use approval results to determine next workflow step'
        ],
        customer_impact:
          'Customers experience a clearer decision process and better documentation of their choices.',
        workflow_stage: 'Decision capture',
        prompt:
          'Explain how customer approval works in the multi-point-inspection API for a dealership service advisor'
      },
      {
        title: 'Close and publish results',
        description: 'Finalize the inspection and share outcomes with internal teams or the customer.',
        business_guidance:
          'This capability completes the inspection lifecycle. It ensures that inspection outcomes are finalized, available for operational follow-through, and usable by both internal dealership teams and customer communication flows.',
        advisor_actions: [
          'Finalize inspection outcomes',
          'Publish or share results where needed',
          'Close the loop between inspection, approval, and execution'
        ],
        customer_impact:
          'Customers receive a clearer completed inspection story and documented results.',
        workflow_stage: 'Inspection completion',
        prompt:
          'Explain how to close and publish results in the multi-point-inspection API for a dealership service advisor'
      }
    ],
    workflow_map: [
      {
        step: 'Start inspection',
        detail: 'Launch MPI workflow tied to intake or RO context',
        prompt:
          'Explain how to start the inspection workflow in the multi-point-inspection API for a dealership service advisor'
      },
      {
        step: 'Capture findings',
        detail: 'Technician records results, notes, and media',
        prompt:
          'Explain how findings are captured in the multi-point-inspection API'
      },
      {
        step: 'Build recommendations',
        detail: 'Advisor prepares customer-facing repair guidance',
        prompt:
          'Explain how recommendations are built in the multi-point-inspection API'
      },
      {
        step: 'Get approval',
        detail: 'Customer decisions are captured and tracked',
        prompt:
          'Explain customer approval flow in the multi-point-inspection API'
      },
      {
        step: 'Move to execution',
        detail: 'Approved work moves into repair-order workflow',
        prompt:
          'Explain how approved work moves into execution from the multi-point-inspection API'
      }
    ]
  }
};

function getBuiltInCapabilityCards(domain) {
  return DOMAIN_CONFIG[domain]?.capability_cards || [];
}

function getBuiltInWorkflowMap(domain) {
  return DOMAIN_CONFIG[domain]?.workflow_map || [];
}

function getBuiltInRichOverview(domain) {
  return DOMAIN_CONFIG[domain]?.rich_overview || `1. What it is
This API supports dealership workflow operations.

2. Core concepts
It provides structured business data and workflow coordination.

3. Main resources
Use schemas, endpoints, and business capabilities to understand the domain.

4. Typical workflow
Use the workflow map to understand how this API supports operations.

5. What to explore next
Inspect capabilities, schemas, and endpoints.`;
}

/* --------------------------------------------------
 * SAFE MCP ENRICHMENT
 * -------------------------------------------------- */

async function safeListSchemas(domain, limit = 6) {
  try {
    return await toolCached('listSchemas', { domain_name: domain, limit });
  } catch {
    return { schemas: [] };
  }
}

async function safeListOperations(domain, limit = 6) {
  try {
    return await toolCached('listOperations', { domain_name: domain, limit });
  } catch {
    return { operations: [] };
  }
}

async function safeGetOverview(domain) {
  try {
    return await toolCached('getApiOverview', { domain_name: domain });
  } catch {
    return null;
  }
}

function deriveEndpointCards(result, domain) {
  return extractOperations(result).slice(0, 6).map((op) => ({
    title: `${op.method || 'GET'} ${op.path || ''}`.trim(),
    description: op.summary || op.description || 'API endpoint',
    prompt: `Explain ${(op.method || 'GET')} ${op.path || ''} for the ${domain || 'current'} API`
  }));
}

function deriveSchemaCards(result, domain) {
  return extractSchemas(result).slice(0, 6).map((s) => ({
    title: s.name,
    description: s.description || 'View schema details',
    prompt: `Show schema ${s.name} in the ${domain || 'current'} API`
  }));
}

/* --------------------------------------------------
 * RESPONSE BUILDERS
 * -------------------------------------------------- */

async function buildCapabilityResponse(domain) {
  const builtInCards = getBuiltInCapabilityCards(domain);
  const workflowMap = getBuiltInWorkflowMap(domain);

  const [ops, schemas] = await Promise.all([
    safeListOperations(domain, 5),
    safeListSchemas(domain, 5)
  ]);

  const endpointCards = deriveEndpointCards(ops, domain);
  const schemaCards = deriveSchemaCards(schemas, domain);

  const answer = `1. Direct answer
Business capabilities for the ${domain} API are shown below.

2. Key details
These capabilities are organized around the dealership workflow and include advisor-facing business meaning, workflow stage, customer impact, and recommended next actions.

3. Useful next steps
Select a capability card to go deeper into business guidance, advisor actions, and related technical context.`;

  return {
    answer,
    sections: extractSections(answer),
    capability_cards: builtInCards,
    endpoint_cards: endpointCards,
    schema_cards: schemaCards,
    workflow_map: workflowMap,
    progressive: endpointCards.length === 0 && schemaCards.length === 0
  };
}

async function buildOverviewResponse(domain, message) {
  const [overview, ops, schemas] = await Promise.all([
    safeGetOverview(domain),
    safeListOperations(domain, 8),
    safeListSchemas(domain, 8)
  ]);

  const builtInOverview = getBuiltInRichOverview(domain);

  let answer;
  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `You are a STAR automotive retail API architect.
The appointment API means dealer vehicle service appointments, not medical scheduling.
The multi-point-inspection API means vehicle inspection workflow.

User request:
${message}

Built-in domain overview:
${builtInOverview}

MCP overview:
${JSON.stringify(overview)}

Operations:
${JSON.stringify(ops)}

Schemas:
${JSON.stringify(schemas)}

Write a rich structured answer that preserves the depth of the built-in domain overview while using MCP data to enrich it where possible.`
      }),
      10000,
      'OpenAI overview generation'
    );

    answer = response.output_text;
  } catch {
    answer = builtInOverview;
  }

  return {
    answer,
    sections: extractSections(answer),
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: deriveEndpointCards(ops, domain),
    schema_cards: deriveSchemaCards(schemas, domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: false
  };
}

async function buildSchemaResponse(domain, message) {
  const schemas = await safeListSchemas(domain, 8);

  let answer;
  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `You are a STAR automotive retail API assistant.

User request:
${message}

Schemas:
${JSON.stringify(schemas)}

Write a concise answer with:
1. Direct answer
2. Key schema details
3. Useful next steps`
      }),
      9000,
      'OpenAI schema answer generation'
    );

    answer = response.output_text;
  } catch {
    answer = `1. Direct answer
Relevant schemas for the ${domain} API are shown in the inspector.

2. Key schema details
Use the schema cards to inspect names and definitions.

3. Useful next steps
Open a schema card or ask for a specific schema by name.`;
  }

  return {
    answer,
    sections: extractSections(answer),
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: [],
    schema_cards: deriveSchemaCards(schemas, domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: false
  };
}

async function buildEndpointResponse(domain, message) {
  const ops = await safeListOperations(domain, 8);

  let answer;
  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `You are a STAR automotive retail API assistant.

User request:
${message}

Operations:
${JSON.stringify(ops)}

Write a concise answer with:
1. Direct answer
2. Key endpoint details
3. Useful next steps`
      }),
      9000,
      'OpenAI endpoint answer generation'
    );

    answer = response.output_text;
  } catch {
    answer = `1. Direct answer
Representative endpoints for the ${domain} API are shown in the inspector.

2. Key endpoint details
Use the endpoint cards to inspect operations and business purpose.

3. Useful next steps
Click an endpoint card or ask for a specific operation.`;
  }

  return {
    answer,
    sections: extractSections(answer),
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: deriveEndpointCards(ops, domain),
    schema_cards: [],
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: false
  };
}

/* --------------------------------------------------
 * ROUTES
 * -------------------------------------------------- */

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'star-ai-intelligence-portal',
    model: MODEL,
    has_openai_key: Boolean(OPENAI_API_KEY),
    has_mcp_server_url: Boolean(MCP_SERVER_URL),
    cache_entries: cache.size
  });
});

app.get('/', (_req, res) => {
  res.json({
    name: 'star-ai-intelligence-portal',
    status: 'ok',
    health: '/health',
    chat: '/api/chat'
  });
});

app.post('/api/chat', async (req, res) => {
  try {
    if (!OPENAI_API_KEY) {
      return res.status(500).json({ error: 'Missing OPENAI_API_KEY' });
    }

    if (!MCP_SERVER_URL) {
      return res.status(500).json({ error: 'Missing MCP_SERVER_URL' });
    }

    const message = req.body?.message;
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'A string "message" is required.' });
    }

    const domain = detectDomain(message);

    if (domain && wantsCapabilities(message)) {
      const result = await buildCapabilityResponse(domain);
      return res.json({
        ...result,
        tool_name: 'business_capability_navigator',
        tool_arguments: { domain_name: domain },
        explore_next: [
          `List operations for the ${domain} API`,
          `List schemas for the ${domain} API`,
          `Show example endpoints for the ${domain} API`
        ]
      });
    }

    if (domain && wantsOverview(message)) {
      const result = await buildOverviewResponse(domain, message);
      return res.json({
        ...result,
        tool_name: 'overview_pipeline',
        tool_arguments: { domain_name: domain },
        explore_next: [
          `Show me the business capabilities of the ${domain} API for a dealership service advisor`,
          `List schemas for the ${domain} API`,
          `Show example endpoints for the ${domain} API`
        ]
      });
    }

    if (domain && wantsSchemas(message)) {
      const result = await buildSchemaResponse(domain, message);
      return res.json({
        ...result,
        tool_name: 'schema_pipeline',
        tool_arguments: { domain_name: domain },
        explore_next: [
          `Show me the business capabilities of the ${domain} API for a dealership service advisor`,
          `Show example endpoints for the ${domain} API`
        ]
      });
    }

    if (domain && wantsEndpoints(message)) {
      const result = await buildEndpointResponse(domain, message);
      return res.json({
        ...result,
        tool_name: 'endpoint_pipeline',
        tool_arguments: { domain_name: domain },
        explore_next: [
          `Show me the business capabilities of the ${domain} API for a dealership service advisor`,
          `List schemas for the ${domain} API`
        ]
      });
    }

    const result = await toolCached('listDomains', {});
    const answer = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `User:
${message}

Data:
${JSON.stringify(result)}

Answer clearly and concisely in the context of STAR automotive APIs.`
      }),
      8000,
      'OpenAI answer generation'
    );

    return res.json({
      answer: answer.output_text,
      sections: extractSections(answer.output_text),
      capability_cards: [],
      endpoint_cards: [],
      schema_cards: [],
      workflow_map: [],
      explore_next: [],
      tool_name: 'listDomains',
      tool_arguments: {}
    });
  } catch (e) {
    const msg = e?.message || 'Unknown error';

    if (msg.includes('Too Many Requests')) {
      return res.status(429).json({
        error: 'The service is being rate limited right now. Please wait a few seconds and try again.'
      });
    }

    if (msg.includes('timed out')) {
      return res.status(504).json({ error: msg });
    }

    return res.status(500).json({ error: msg });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`STAR matched-pair server running on ${PORT}`);
});
