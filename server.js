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

function isRateLimitErrorMessage(msg = '') {
  const m = String(msg).toLowerCase();
  return (
    m.includes('too many requests') ||
    m.includes('rate limit') ||
    m.includes('rate-limited') ||
    m.includes('being rate limited') ||
    m.includes('"code":429') ||
    m.includes('429')
  );
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

function normalizeAudience(audience = '') {
  const a = String(audience).toLowerCase();
  if (a === 'technical') return 'technical';
  if (a === 'architecture') return 'architecture';
  return 'business';
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
    business_overview: `1. What it is
The Appointment API represents the dealership service scheduling and intake layer. It helps coordinate customer bookings, requested services, timing expectations, and operational readiness before the vehicle enters the service lane.

2. Core concepts
The API centers on the service appointment as the planning object that connects customer intent, vehicle context, dealership availability, and requested work.

3. Main resources
Typical resources include the appointment itself, requested services, customer and vehicle references, timing details, and status information needed by dealership systems.

4. Typical workflow
A customer books service, the dealership validates the request, the advisor confirms timing and service intent, and the service lane uses the appointment context to prepare intake and execution.

5. What to explore next
Look at business capabilities, representative endpoints, and schemas related to scheduling, intake, and service preparation.`,

    technical_overview: `1. What it is
The Appointment API is the scheduling and intake domain for dealership service operations. It models appointment creation, requested service context, party and vehicle references, timing, and workflow state.

2. Technical model
A typical implementation treats Appointment as a bounded-context resource with references to customer, vehicle, requested services, and scheduling metadata. The API helps move from customer intent into executable service-lane context.

3. Resource families
Typical technical resources include appointment entities, request details, time windows, references, and status fields needed for intake orchestration.

4. Integration concerns
Technical consumers care about identifiers, party and vehicle references, status transitions, and how appointment context flows into downstream service systems.

5. What to explore next
Inspect schema cards, endpoint cards, required fields, and object relationships.`,

    architecture_overview: `1. Bounded context role
The Appointment API sits at the front of the dealership service journey. It owns service-booking and pre-intake coordination concerns.

2. Boundary
Its responsibility is not repair execution or detailed inspection results. Its role is to create the operational context that downstream domains consume.

3. Cross-domain relationships
Appointment typically connects to customer, vehicle, scheduling, intake, and downstream service execution contexts.

4. Architectural significance
This domain is valuable because it separates scheduling intent from execution concerns while still providing the references needed for downstream orchestration.

5. What to explore next
Review bounded-context relationships, key references, and how Appointment feeds inspection and repair workflows.`,

    capability_cards: [
      {
        title: 'Schedule appointment',
        description: 'Create and manage service appointments for customers and vehicles.',
        business_guidance:
          'This capability covers the initial booking process and represents the front door into service operations.',
        advisor_actions: [
          'Review requested services and timing',
          'Confirm customer and vehicle context',
          'Make sure appointment data supports intake readiness'
        ],
        customer_impact:
          'Customers get a clearer booking experience and more accurate service expectations.',
        workflow_stage: 'Scheduling',
        technical_notes:
          'Developers should focus on appointment identifiers, requested service structures, time windows, and party or vehicle references.',
        data_model_notes:
          'The core model typically includes appointment identity, service intent, scheduling metadata, and supporting references.',
        integration_notes:
          'This capability commonly feeds service-lane preparation, intake workflows, and downstream service domains.',
        prompt:
          'Explain how to schedule and manage appointments in the Appointment API'
      },
      {
        title: 'Confirm service needs',
        description: 'Validate requested services, timing, and customer expectations.',
        business_guidance:
          'This capability helps the dealership turn a raw booking into an operationally useful service request.',
        advisor_actions: [
          'Validate requested work',
          'Check timing and operational fit',
          'Reduce ambiguity before service intake'
        ],
        customer_impact:
          'Customers receive more accurate service planning and fewer surprises at intake.',
        workflow_stage: 'Pre-intake validation',
        technical_notes:
          'Technical consumers should examine how requested services, notes, and appointment state are represented and updated.',
        data_model_notes:
          'This area often involves validation rules, optional versus required fields, and how pre-intake details are normalized.',
        integration_notes:
          'This step often influences downstream systems that consume intake-ready appointment context.',
        prompt:
          'Explain how a service advisor confirms service needs using the Appointment API'
      },
      {
        title: 'Prepare service lane',
        description: 'Use appointment context to prepare the dealership service workflow.',
        business_guidance:
          'This capability connects the appointment to the dealership operating model and helps the service lane prepare before the customer arrives.',
        advisor_actions: [
          'Review upcoming appointments',
          'Use appointment data to prepare intake',
          'Coordinate service-lane readiness'
        ],
        customer_impact:
          'Customers experience smoother intake and better operational coordination.',
        workflow_stage: 'Service preparation',
        technical_notes:
          'Technical teams should look at how appointment data is exposed for operational consumers and how readiness-related status is modeled.',
        data_model_notes:
          'Key elements usually include schedule context, requested services, and references needed by intake systems.',
        integration_notes:
          'This capability commonly interacts with scheduling, check-in, and service execution contexts.',
        prompt:
          'Explain how the Appointment API helps prepare the service lane'
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
    ],

    schema_cards: [
      {
        title: 'Appointment',
        description: 'Core appointment entity for dealership service scheduling.',
        prompt: 'Show schema Appointment in the Appointment API'
      },
      {
        title: 'RequestedService',
        description: 'Represents requested service work tied to an appointment.',
        prompt: 'Show schema RequestedService in the Appointment API'
      },
      {
        title: 'AppointmentStatus',
        description: 'Represents appointment lifecycle or state.',
        prompt: 'Show schema AppointmentStatus in the Appointment API'
      },
      {
        title: 'VehicleReference',
        description: 'Vehicle identity and linkage used by the appointment domain.',
        prompt: 'Show schema VehicleReference in the Appointment API'
      },
      {
        title: 'PartyReference',
        description: 'Customer or related party linkage used in the appointment domain.',
        prompt: 'Show schema PartyReference in the Appointment API'
      },
      {
        title: 'TimeSlot',
        description: 'Scheduling or timing structure for appointment planning.',
        prompt: 'Show schema TimeSlot in the Appointment API'
      }
    ],

    endpoint_cards: [
      {
        title: 'GET /appointments',
        description: 'List or search appointments.',
        prompt: 'Explain GET /appointments for the Appointment API'
      },
      {
        title: 'POST /appointments',
        description: 'Create a new appointment.',
        prompt: 'Explain POST /appointments for the Appointment API'
      },
      {
        title: 'GET /appointments/{id}',
        description: 'Retrieve a specific appointment.',
        prompt: 'Explain GET /appointments/{id} for the Appointment API'
      },
      {
        title: 'PATCH /appointments/{id}',
        description: 'Update an existing appointment.',
        prompt: 'Explain PATCH /appointments/{id} for the Appointment API'
      },
      {
        title: 'GET /appointments/{id}/requested-services',
        description: 'View requested services tied to an appointment.',
        prompt: 'Explain GET /appointments/{id}/requested-services for the Appointment API'
      }
    ]
  },

  'multi-point-inspection': {
    business_overview: `1. What it is
The Multi-Point Inspection API is a dealership service workflow domain that supports structured vehicle health checks. It covers the lifecycle from launching an inspection through technician findings, recommendations, customer decisions, and final completion.

2. Core concepts
The domain revolves around the inspection itself, what is being inspected, what was found, and what action should be recommended or approved.

3. Main resources
Typical resources include the inspection record, findings, condition evidence, media, recommendations, approval outcomes, and workflow status.

4. Typical workflow
An inspection is started from service intake or repair-order context. The technician records findings and supporting evidence. The advisor reviews the results, prepares recommendations, communicates with the customer, captures approvals or declines, and moves approved work into execution.

5. What to explore next
Review business capabilities, workflow stages, representative endpoints, and schemas related to findings, recommendations, approvals, and inspection completion.`,

    technical_overview: `1. What it is
The Multi-Point Inspection API is the structured inspection domain for dealership vehicle health checks. It models the inspection lifecycle, findings, media, recommendations, and decision outcomes.

2. Technical model
Technical consumers should think in terms of inspection entities, finding collections, evidence/media attachments, recommendation structures, and approval or completion states.

3. Resource families
Common resource families include inspection records, line-item findings, media or attachments, recommendation artifacts, customer decision objects, and workflow status indicators.

4. Integration concerns
Developers care about how inspection context is created, how findings are represented, how recommendations connect to actionable work, and how approval outcomes are captured for downstream execution.

5. What to explore next
Inspect endpoints, schema groups, object relationships, identifiers, and how inspection data flows into service execution contexts.`,

    architecture_overview: `1. Bounded context role
The Multi-Point Inspection API owns structured inspection and recommendation workflow within the dealership service domain.

2. Boundary
Its responsibility is not full appointment scheduling or full repair-order execution. It sits between intake/service context and downstream execution decisions.

3. Cross-domain relationships
MPI usually depends on upstream intake, visit, or repair-order context and feeds downstream recommendation, approval, and execution processes.

4. Architectural significance
This domain is important because it transforms technician-level observations into customer-facing service decisions while preserving operational traceability.

5. What to explore next
Review relationships between inspection, findings, recommendations, approvals, and execution-oriented downstream systems.`,

    capability_cards: [
      {
        title: 'Start inspection workflow',
        description: 'Begin the MPI process from intake, appointment context, or repair-order workflow.',
        business_guidance:
          'This capability launches the inspection lifecycle and provides technicians and advisors a common workflow anchor.',
        advisor_actions: [
          'Initiate inspection from service context',
          'Ensure the inspection is tied to the right vehicle and visit',
          'Use the inspection as the operational anchor for downstream findings'
        ],
        customer_impact:
          'Customers benefit from a structured and traceable inspection process.',
        workflow_stage: 'Inspection initiation',
        technical_notes:
          'Technical consumers should examine how inspection identity is created, what upstream references are required, and how initialization state is modeled.',
        data_model_notes:
          'The core model usually includes inspection identifiers, visit or vehicle references, workflow state, and optional linkage to upstream intake context.',
        integration_notes:
          'This capability often bridges appointment, visit, or repair-order context into the inspection domain.',
        prompt:
          'Explain how to start the inspection workflow in the multi-point-inspection API'
      },
      {
        title: 'Track inspection progress',
        description: 'Monitor inspection state, technician progress, and workflow status.',
        business_guidance:
          'This capability gives advisors visibility into where the inspection stands and when customer-facing follow-up can begin.',
        advisor_actions: [
          'Monitor inspection lifecycle status',
          'Coordinate timing between technician work and advisor communication',
          'Use progress visibility to manage service-lane flow'
        ],
        customer_impact:
          'Customers receive more timely updates and better expectation management.',
        workflow_stage: 'Inspection monitoring',
        technical_notes:
          'Developers should inspect status models, progress-related fields, and operation families that expose lifecycle transitions.',
        data_model_notes:
          'This area often includes workflow status, timestamps, completion indicators, and state transitions.',
        integration_notes:
          'Inspection progress is often consumed by advisor tools, dashboards, and customer communication layers.',
        prompt:
          'Explain how to track inspection progress in the multi-point-inspection API'
      },
      {
        title: 'Review findings and media',
        description: 'Review technician findings, notes, photos, and condition evidence.',
        business_guidance:
          'This capability is where raw technician observations become advisor-usable business information.',
        advisor_actions: [
          'Review condition findings',
          'Use media and notes as supporting evidence',
          'Translate technical observations into customer-facing explanation'
        ],
        customer_impact:
          'Customers receive clearer, evidence-based explanations of vehicle condition.',
        workflow_stage: 'Findings review',
        technical_notes:
          'Technical teams should examine how findings are represented, how media links are attached, and how condition evidence is structured.',
        data_model_notes:
          'The model often includes finding identity, severity or status, notes, media associations, and structured inspection result fields.',
        integration_notes:
          'This capability commonly feeds recommendation generation, customer communication, and archival workflows.',
        prompt:
          'Explain how to review findings and media in the multi-point-inspection API'
      },
      {
        title: 'Prepare recommendations',
        description: 'Turn findings into advisor-ready repair recommendations and next steps.',
        business_guidance:
          'This capability converts inspection results into action and prepares customer-ready proposals tied to vehicle condition and service priorities.',
        advisor_actions: [
          'Turn findings into recommended work',
          'Organize repair guidance by business relevance',
          'Prepare customer-facing next steps'
        ],
        customer_impact:
          'Customers receive clearer recommendations and understand why work is being proposed.',
        workflow_stage: 'Recommendation building',
        technical_notes:
          'Technical consumers should inspect how recommendations relate to findings, how recommended actions are modeled, and whether downstream execution references exist.',
        data_model_notes:
          'This layer often contains structured recommendation entities, finding links, optional pricing context, and approval-ready outputs.',
        integration_notes:
          'This capability may connect inspection findings to repair planning, pricing, parts, or service execution contexts.',
        prompt:
          'Explain how to prepare recommendations in the multi-point-inspection API'
      },
      {
        title: 'Capture customer approval',
        description: 'Record customer decisions, approvals, and communication outcomes.',
        business_guidance:
          'This capability captures the business outcome of the advisor-customer interaction and determines what work moves forward.',
        advisor_actions: [
          'Record approvals and declines',
          'Track customer communication outcomes',
          'Use approval results to determine next workflow step'
        ],
        customer_impact:
          'Customers experience a clearer decision process and better documentation of their choices.',
        workflow_stage: 'Decision capture',
        technical_notes:
          'Technical teams should inspect approval state, decline/approve semantics, and how decision artifacts are attached to recommendation structures.',
        data_model_notes:
          'Common model elements include approval outcomes, timestamps, decision notes, and references to recommendation or finding entities.',
        integration_notes:
          'Approval outcomes are often consumed by execution, customer-history, and reporting flows.',
        prompt:
          'Explain how customer approval works in the multi-point-inspection API'
      },
      {
        title: 'Close and publish results',
        description: 'Finalize the inspection and share outcomes with internal teams or the customer.',
        business_guidance:
          'This capability completes the inspection lifecycle and ensures that finalized outcomes are available for follow-through and communication.',
        advisor_actions: [
          'Finalize inspection outcomes',
          'Publish or share results where needed',
          'Close the loop between inspection, approval, and execution'
        ],
        customer_impact:
          'Customers receive a clearer completed inspection story and documented results.',
        workflow_stage: 'Inspection completion',
        technical_notes:
          'Technical consumers should inspect completion state, publication or output semantics, and how finalized inspection data is represented.',
        data_model_notes:
          'This layer typically includes closed-state indicators, final result artifacts, and references to downstream usage or publication.',
        integration_notes:
          'Completion often feeds archival, reporting, customer-history, and execution-oriented systems.',
        prompt:
          'Explain how to close and publish results in the multi-point-inspection API'
      }
    ],

    workflow_map: [
      {
        step: 'Start inspection',
        detail: 'Launch MPI workflow tied to intake or RO context',
        prompt:
          'Explain how to start the inspection workflow in the multi-point-inspection API'
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
    ],

    schema_cards: [
      {
        title: 'Inspection',
        description: 'Core inspection entity representing the inspection lifecycle.',
        prompt: 'Show schema Inspection in the multi-point-inspection API'
      },
      {
        title: 'Finding',
        description: 'Represents an inspection finding, condition, or observed result.',
        prompt: 'Show schema Finding in the multi-point-inspection API'
      },
      {
        title: 'Recommendation',
        description: 'Represents advisor-ready or technician-derived recommended work.',
        prompt: 'Show schema Recommendation in the multi-point-inspection API'
      },
      {
        title: 'Approval',
        description: 'Captures customer approval or decline decisions.',
        prompt: 'Show schema Approval in the multi-point-inspection API'
      },
      {
        title: 'InspectionMedia',
        description: 'Represents photos, videos, or other evidence attached to findings.',
        prompt: 'Show schema InspectionMedia in the multi-point-inspection API'
      },
      {
        title: 'InspectionStatus',
        description: 'Represents workflow or lifecycle state of an inspection.',
        prompt: 'Show schema InspectionStatus in the multi-point-inspection API'
      }
    ],

    endpoint_cards: [
      {
        title: 'GET /inspections',
        description: 'List or search inspections.',
        prompt: 'Explain GET /inspections for the multi-point-inspection API'
      },
      {
        title: 'POST /inspections',
        description: 'Create or start an inspection.',
        prompt: 'Explain POST /inspections for the multi-point-inspection API'
      },
      {
        title: 'GET /inspections/{id}',
        description: 'Retrieve a specific inspection.',
        prompt: 'Explain GET /inspections/{id} for the multi-point-inspection API'
      },
      {
        title: 'GET /inspections/{id}/findings',
        description: 'Retrieve findings tied to an inspection.',
        prompt: 'Explain GET /inspections/{id}/findings for the multi-point-inspection API'
      },
      {
        title: 'POST /inspections/{id}/recommendations',
        description: 'Create or record recommendations for an inspection.',
        prompt: 'Explain POST /inspections/{id}/recommendations for the multi-point-inspection API'
      },
      {
        title: 'POST /inspections/{id}/approvals',
        description: 'Capture approval or decline outcomes.',
        prompt: 'Explain POST /inspections/{id}/approvals for the multi-point-inspection API'
      }
    ]
  }
};

function getAudienceOverview(domain, audience) {
  const cfg = DOMAIN_CONFIG[domain];
  if (!cfg) return 'No overview available.';
  if (audience === 'technical') return cfg.technical_overview;
  if (audience === 'architecture') return cfg.architecture_overview;
  return cfg.business_overview;
}

function getBuiltInCapabilityCards(domain) {
  return DOMAIN_CONFIG[domain]?.capability_cards || [];
}

function getBuiltInWorkflowMap(domain) {
  return DOMAIN_CONFIG[domain]?.workflow_map || [];
}

function getBuiltInSchemaCards(domain) {
  return DOMAIN_CONFIG[domain]?.schema_cards || [];
}

function getBuiltInEndpointCards(domain) {
  return DOMAIN_CONFIG[domain]?.endpoint_cards || [];
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

async function buildCapabilityResponse(domain, audience) {
  const builtInCards = getBuiltInCapabilityCards(domain);
  const workflowMap = getBuiltInWorkflowMap(domain);

  const [ops, schemas] = await Promise.all([
    safeListOperations(domain, 5),
    safeListSchemas(domain, 5)
  ]);

  const liveEndpointCards = deriveEndpointCards(ops, domain);
  const liveSchemaCards = deriveSchemaCards(schemas, domain);

  const endpointCards = liveEndpointCards.length
    ? liveEndpointCards
    : getBuiltInEndpointCards(domain);

  const schemaCards = liveSchemaCards.length
    ? liveSchemaCards
    : getBuiltInSchemaCards(domain);

  let intro;
  if (audience === 'technical') {
    intro = `1. Direct answer
Technical capability views for the ${domain} API are shown below.

2. Key details
Each capability includes technical notes, data model notes, and integration notes in addition to business meaning.

3. Useful next steps
Select a capability card to inspect technical details and implementation context.`;
  } else if (audience === 'architecture') {
    intro = `1. Direct answer
Architecture-oriented capability views for the ${domain} API are shown below.

2. Key details
Each capability includes context boundaries, data-model implications, and integration implications relevant to architecture work.

3. Useful next steps
Select a capability card to inspect architecture-oriented details and dependencies.`;
  } else {
    intro = `1. Direct answer
Business capabilities for the ${domain} API are shown below.

2. Key details
These capabilities are organized around the dealership workflow and include advisor-facing meaning, workflow stage, and customer impact.

3. Useful next steps
Select a capability card to go deeper into business guidance and next actions.`;
  }

  return {
    answer: intro,
    sections: extractSections(intro),
    capability_cards: builtInCards,
    endpoint_cards: endpointCards,
    schema_cards: schemaCards,
    workflow_map: workflowMap,
    progressive: liveEndpointCards.length === 0 || liveSchemaCards.length === 0
  };
}

async function buildOverviewResponse(domain, message, audience) {
  const [overview, ops, schemas] = await Promise.all([
    safeGetOverview(domain),
    safeListOperations(domain, 8),
    safeListSchemas(domain, 8)
  ]);

  const builtInOverview = getAudienceOverview(domain, audience);

  let answer;
  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `You are a STAR automotive retail API architect.
Audience: ${audience}
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

  const liveEndpointCards = deriveEndpointCards(ops, domain);
  const liveSchemaCards = deriveSchemaCards(schemas, domain);

  return {
    answer,
    sections: extractSections(answer),
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: liveEndpointCards.length ? liveEndpointCards : getBuiltInEndpointCards(domain),
    schema_cards: liveSchemaCards.length ? liveSchemaCards : getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: liveEndpointCards.length === 0 || liveSchemaCards.length === 0
  };
}

async function buildSchemaResponse(domain, message, audience) {
  const schemas = await safeListSchemas(domain, 8);
  const liveSchemaCards = deriveSchemaCards(schemas, domain);
  const schemaCards = liveSchemaCards.length ? liveSchemaCards : getBuiltInSchemaCards(domain);

  let answer;
  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `You are a STAR automotive retail API assistant.
Audience: ${audience}

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
Representative schemas for the ${domain} API are shown in the inspector.

2. Key schema details
These schema cards represent the primary object families and data structures that technical and business users typically need first.

3. Useful next steps
Open a schema card, switch to Technical mode for schema-oriented reading, or ask for a specific schema by name.`;
  }

  return {
    answer,
    sections: extractSections(answer),
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: [],
    schema_cards: schemaCards,
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: liveSchemaCards.length === 0
  };
}

async function buildEndpointResponse(domain, message, audience) {
  const ops = await safeListOperations(domain, 5);
  const liveEndpointCards = deriveEndpointCards(ops, domain);
  const endpointCards = liveEndpointCards.length ? liveEndpointCards : getBuiltInEndpointCards(domain);

  let answer;
  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `You are a STAR automotive retail API assistant.
Audience: ${audience}

User request:
${message}

Operations:
${JSON.stringify(ops)}

Write a concise answer with:
1. Direct answer
2. Key endpoint details
3. Useful next steps`
      }),
      15000,
      'OpenAI endpoint answer generation'
    );

    answer = response.output_text;
  } catch {
    answer = `1. Direct answer
Representative endpoints for the ${domain} API are shown in the inspector.

2. Key endpoint details
These endpoints represent the main operation families for the domain and can be used to understand how clients interact with the API.

3. Useful next steps
Inspect the endpoint cards, switch to Technical mode for implementation-oriented reading, or ask for a specific operation, method, or path.`;
  }

  return {
    answer,
    sections: extractSections(answer),
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: endpointCards,
    schema_cards: [],
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: liveEndpointCards.length === 0
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
    const audience = normalizeAudience(req.body?.audience || 'business');

    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'A string "message" is required.' });
    }

    const domain = detectDomain(message);

    if (domain && wantsCapabilities(message)) {
      const result = await buildCapabilityResponse(domain, audience);
      return res.json({
        ...result,
        audience,
        tool_name: 'business_capability_navigator',
        tool_arguments: { domain_name: domain, audience },
        explore_next: [
          `List operations for the ${domain} API`,
          `List schemas for the ${domain} API`,
          `Show example endpoints for the ${domain} API`
        ]
      });
    }

    if (domain && wantsOverview(message)) {
      const result = await buildOverviewResponse(domain, message, audience);
      return res.json({
        ...result,
        audience,
        tool_name: 'overview_pipeline',
        tool_arguments: { domain_name: domain, audience },
        explore_next: [
          `Show me the business capabilities of the ${domain} API`,
          `List schemas for the ${domain} API`,
          `Show example endpoints for the ${domain} API`
        ]
      });
    }

    if (domain && wantsSchemas(message)) {
      const result = await buildSchemaResponse(domain, message, audience);
      return res.json({
        ...result,
        audience,
        tool_name: 'schema_pipeline',
        tool_arguments: { domain_name: domain, audience },
        explore_next: [
          `Show me the business capabilities of the ${domain} API`,
          `Show example endpoints for the ${domain} API`
        ]
      });
    }

    if (domain && wantsEndpoints(message)) {
      const result = await buildEndpointResponse(domain, message, audience);
      return res.json({
        ...result,
        audience,
        tool_name: 'endpoint_pipeline',
        tool_arguments: { domain_name: domain, audience },
        explore_next: [
          `Show me the business capabilities of the ${domain} API`,
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

Audience:
${audience}

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
      audience,
      capability_cards: [],
      endpoint_cards: [],
      schema_cards: [],
      workflow_map: [],
      explore_next: [],
      tool_name: 'listDomains',
      tool_arguments: { audience }
    });
  } catch (e) {
    const msg = e?.message || 'Unknown error';

    if (isRateLimitErrorMessage(msg)) {
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
  console.log(`STAR audience-aware server running on ${PORT}`);
});
