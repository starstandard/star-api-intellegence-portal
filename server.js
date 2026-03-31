import 'dotenv/config';
import express from 'express';
import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const PORT = Number(process.env.PORT || 3000);
const MODEL = process.env.OPENAI_MODEL || 'gpt-5';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';

const client = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

/* -------------------------
 * UTILITIES
 * ------------------------- */

const now = () => Date.now();

function requestId() {
  return crypto.randomUUID();
}

function toSections(text) {
  const value = String(text || '').trim();
  if (!value) return [];
  return [{ title: 'Direct answer', body: value }];
}

function structuredSectionsFromNumberedText(text) {
  const value = String(text || '').trim();
  if (!value) return [];

  const lines = value.split('\n');
  const sections = [];
  let current = null;

  for (const raw of lines) {
    const line = raw.trim();

    if (!line) {
      if (current) current.body.push('');
      continue;
    }

    const match = line.match(/^(\d+)\.\s+(.+)$/);
    if (match) {
      if (current) {
        sections.push({
          title: current.title,
          body: current.body.join('\n').trim()
        });
      }
      current = { title: match[2], body: [] };
    } else {
      if (!current) current = { title: 'Direct answer', body: [] };
      current.body.push(line);
    }
  }

  if (current) {
    sections.push({
      title: current.title,
      body: current.body.join('\n').trim()
    });
  }

  return sections.length ? sections : toSections(value);
}

async function withTimeout(promise, ms, label = 'Operation') {
  let timeout;
  const timer = new Promise((_, reject) => {
    timeout = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });

  try {
    return await Promise.race([promise, timer]);
  } finally {
    clearTimeout(timeout);
  }
}

async function openaiCall(input, retries = 1) {
  if (!client) {
    throw new Error('OpenAI client is not configured');
  }

  for (let i = 0; i <= retries; i += 1) {
    try {
      return await withTimeout(
        client.responses.create({
          model: MODEL,
          input
        }),
        20000,
        'OpenAI answer generation'
      );
    } catch (error) {
      if (i === retries) throw error;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  throw new Error('Unexpected OpenAI retry failure');
}

/* -------------------------
 * LOAD OPENAPI
 * ------------------------- */

const ROOT = process.cwd();

const OPENAPI_PATHS = {
  appointment:
    process.env.APPOINTMENT_OPENAPI_SPEC_PATH ||
    path.join(ROOT, 'openapi', 'appointment-api.yaml'),
  'multi-point-inspection':
    process.env.MPI_OPENAPI_SPEC_PATH ||
    path.join(ROOT, 'openapi', 'multi-point-inspection-api.yaml')
};

function loadSpec(file) {
  try {
    if (!fs.existsSync(file)) return null;
    return yaml.load(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
}

const REGISTRY = {
  appointment: loadSpec(OPENAPI_PATHS.appointment),
  'multi-point-inspection': loadSpec(OPENAPI_PATHS['multi-point-inspection'])
};

/* -------------------------
 * OPENAPI HELPERS
 * ------------------------- */

function detectDomain(msg = '') {
  const m = String(msg).toLowerCase();
  if (m.includes('inspection') || m.includes('mpi')) return 'multi-point-inspection';
  if (m.includes('appointment')) return 'appointment';
  return null;
}

function getSchemas(domain) {
  return REGISTRY[domain]?.components?.schemas || {};
}

function getOperations(domain) {
  const spec = REGISTRY[domain];
  if (!spec?.paths) return [];

  const ops = [];
  for (const [routePath, obj] of Object.entries(spec.paths)) {
    for (const method of ['get', 'post', 'put', 'patch', 'delete']) {
      if (obj[method]) {
        ops.push({
          method: method.toUpperCase(),
          path: routePath,
          summary: obj[method].summary || '',
          description: obj[method].description || '',
          parameters: obj[method].parameters || [],
          requestBody: obj[method].requestBody || null,
          responses: obj[method].responses || {}
        });
      }
    }
  }

  return ops;
}

function getOperationByMethodAndPath(domain, method, routePath) {
  const spec = REGISTRY[domain];
  if (!spec?.paths?.[routePath]) return null;
  return spec.paths[routePath][String(method).toLowerCase()] || null;
}

function normalizeEndpointPrompt(message = '') {
  const match = String(message).match(/(GET|POST|PUT|PATCH|DELETE)\s+\/[A-Za-z0-9/_{}-]+/i);
  return match ? match[0] : null;
}

function schemaNameFromRef(ref = '') {
  const match = String(ref).match(/#\/components\/schemas\/(.+)$/);
  return match?.[1] || null;
}

function collectSchemaRefs(value, acc = new Set()) {
  if (!value || typeof value !== 'object') return acc;

  if (typeof value.$ref === 'string') {
    const name = schemaNameFromRef(value.$ref);
    if (name) acc.add(name);
  }

  for (const child of Object.values(value)) {
    if (child && typeof child === 'object') {
      collectSchemaRefs(child, acc);
    }
  }

  return acc;
}

function buildExampleFromSchema(schema, schemas, depth = 0) {
  if (!schema || depth > 4) return null;

  if (schema.example !== undefined) return schema.example;

  if (schema.$ref) {
    const name = schemaNameFromRef(schema.$ref);
    const target = schemas[name];
    return target ? buildExampleFromSchema(target, schemas, depth + 1) : null;
  }

  if (schema.enum?.length) return schema.enum[0];

  if (schema.type === 'object' || schema.properties) {
    const out = {};
    for (const [key, value] of Object.entries(schema.properties || {})) {
      const built = buildExampleFromSchema(value, schemas, depth + 1);
      out[key] = built !== null ? built : '<value>';
    }
    return out;
  }

  if (schema.type === 'array') {
    const item = buildExampleFromSchema(schema.items, schemas, depth + 1);
    return [item !== null ? item : '<item>'];
  }

  switch (schema.type) {
    case 'string':
      if (schema.format === 'date-time') return '2026-03-31T12:00:00Z';
      if (schema.format === 'date') return '2026-03-31';
      return '<string>';
    case 'integer':
    case 'number':
      return 0;
    case 'boolean':
      return true;
    default:
      return null;
  }
}

function firstJsonContent(content) {
  if (!content || typeof content !== 'object') return null;
  return (
    content['application/json'] ||
    content['application/*+json'] ||
    Object.values(content)[0] ||
    null
  );
}

function extractEndpointDetail(domain, method, routePath) {
  const op = getOperationByMethodAndPath(domain, method, routePath);
  if (!op) return null;

  const schemas = getSchemas(domain);
  const parameters = op.parameters || [];

  const requestContent = firstJsonContent(op.requestBody?.content);
  const responses = op.responses || {};
  const preferredResponse =
    responses['200'] ||
    responses['201'] ||
    responses['202'] ||
    Object.values(responses)[0];

  const responseContent = firstJsonContent(preferredResponse?.content);

  const relatedNames = new Set();
  collectSchemaRefs(op, relatedNames);
  if (requestContent?.schema) collectSchemaRefs(requestContent.schema, relatedNames);
  if (responseContent?.schema) collectSchemaRefs(responseContent.schema, relatedNames);

  const relatedSchemaCards = Array.from(relatedNames).map((name) => ({
    title: name,
    description: schemas[name]?.description || 'Related OpenAPI schema',
    prompt: `Show schema ${name} in the ${domain} API`
  }));

  let requestExample = null;
  if (requestContent?.example !== undefined) {
    requestExample = requestContent.example;
  } else if (requestContent?.examples && typeof requestContent.examples === 'object') {
    const first = Object.values(requestContent.examples)[0];
    requestExample = first?.value ?? null;
  } else if (requestContent?.schema) {
    requestExample = buildExampleFromSchema(requestContent.schema, schemas);
  }

  let responseExample = null;
  if (responseContent?.example !== undefined) {
    responseExample = responseContent.example;
  } else if (responseContent?.examples && typeof responseContent.examples === 'object') {
    const first = Object.values(responseContent.examples)[0];
    responseExample = first?.value ?? null;
  } else if (responseContent?.schema) {
    responseExample = buildExampleFromSchema(responseContent.schema, schemas);
  }

  return {
    summary: op.summary || '',
    description: op.description || '',
    endpoint_parameters: parameters,
    related_schema_cards: relatedSchemaCards,
    request_example: requestExample,
    response_example: responseExample,
    raw_openapi: op
  };
}

function getCapabilityCards(domain) {
  if (domain === 'multi-point-inspection') {
    return [
      {
        title: 'Inspection overview',
        description: 'Understand the role of the Multi-Point Inspection API in dealership workflows.',
        prompt: 'Give me a rich description of the Multi-Point Inspection API'
      },
      {
        title: 'Inspection workflow',
        description: 'Follow the lifecycle from inspection creation through findings and recommendations.',
        prompt: 'Explain how to start the inspection workflow in the Multi-Point Inspection API'
      },
      {
        title: 'Customer approval',
        description: 'Review how recommendations and decisions support downstream service execution.',
        prompt: 'Explain customer approval in the Multi-Point Inspection API'
      }
    ];
  }

  if (domain === 'appointment') {
    return [
      {
        title: 'Appointment overview',
        description: 'Understand the role of the Appointment API in service scheduling and intake.',
        prompt: 'Give me a rich description of the Appointment API'
      },
      {
        title: 'Scheduling workflow',
        description: 'Follow the lifecycle from appointment creation to service-lane preparation.',
        prompt: 'Explain the appointment workflow in the Appointment API'
      },
      {
        title: 'Requested service context',
        description: 'Review how requested service details support dealership coordination.',
        prompt: 'Explain requested service context in the Appointment API'
      }
    ];
  }

  return [];
}

/* -------------------------
 * DEMO MODE CONTENT
 * ------------------------- */

function getOverview(domain) {
  if (domain === 'multi-point-inspection') {
    return `1. Direct answer
The Multi-Point Inspection API supports structured vehicle inspections in dealership service operations.

2. Core responsibilities
It manages inspections, findings, media, recommendations, and approval-related outcomes.

3. Why it matters
It connects technician observations to advisor communication and downstream service execution.`;
  }

  if (domain === 'appointment') {
    return `1. Direct answer
The Appointment API supports dealership service scheduling and intake coordination.

2. Core responsibilities
It manages appointments, requested service context, timing, and visit preparation.

3. Why it matters
It provides the upstream service workflow context needed before inspection and repair activity begins.`;
  }

  return null;
}

function getWorkflowAnswer(domain) {
  if (domain === 'multi-point-inspection') {
    return `1. Direct answer
The inspection workflow begins by creating an Inspection tied to a vehicle service visit or repair order.

2. Key details
Technicians capture findings and media, then advisors review results and recommendations with the customer.

3. Why it matters
This workflow supports transparency, approval decisions, and downstream service execution.`;
  }

  if (domain === 'appointment') {
    return `1. Direct answer
The appointment workflow begins by creating a service appointment for a customer and vehicle.

2. Key details
The appointment captures requested services, timing, and visit context before downstream operational work begins.

3. Why it matters
It helps the dealership coordinate service readiness and customer expectations.`;
  }

  return null;
}

function getCapabilityAnswer(domain, message) {
  const lower = String(message).toLowerCase();

  if (domain === 'multi-point-inspection') {
    if (lower.includes('customer approval')) {
      return `1. Direct answer
Customer approval in the Multi-Point Inspection API represents the decision point after findings and recommendations are reviewed.

2. Key details
It connects inspection outputs to accepted or declined service actions and helps advisors move from technical findings to customer decisions.

3. Why it matters
It is the workflow bridge between inspection intelligence and downstream execution.`;
    }
  }

  if (domain === 'appointment') {
    if (lower.includes('requested service context')) {
      return `1. Direct answer
Requested service context in the Appointment API captures what work the customer wants performed.

2. Key details
It helps the dealership align scheduling, intake, and preparation before the visit begins.

3. Why it matters
It gives downstream workflows the service intent needed for operational readiness.`;
    }
  }

  return null;
}

function getBuiltInEndpointDetail(endpoint) {
  if (!endpoint) return null;
  const key = endpoint.toLowerCase();

  if (key.includes('/inspections') && key.includes('/findings')) {
    return {
      answer: `1. Direct answer
GET /inspections/{id}/findings retrieves the findings recorded for a specific inspection.

2. Key endpoint details
Those findings represent technician observations, condition results, notes, and possibly related media.

3. Why it matters
This is a key read path for advisor review, customer communication, and recommendation building.`,
      parameters: [
        { name: 'id', in: 'path', required: true, type: 'string' }
      ]
    };
  }

  return null;
}

/* -------------------------
 * UI + HEALTH
 * ------------------------- */

app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/health', (_req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'star-ai-intelligence-portal',
    version: 'demo-mode-v3'
  });
});

/* -------------------------
 * VERSION + OPENAPI ROUTES
 * ------------------------- */

app.get('/api/version', (_req, res) => {
  res.json({
    version: 'demo-mode-v3',
    domains: {
      appointment: {
        openapi_loaded: !!REGISTRY.appointment,
        openapi_source: OPENAPI_PATHS.appointment
      },
      'multi-point-inspection': {
        openapi_loaded: !!REGISTRY['multi-point-inspection'],
        openapi_source: OPENAPI_PATHS['multi-point-inspection']
      }
    }
  });
});

app.get('/api/openapi-index', (req, res) => {
  const domain = String(req.query.domain || '');
  const spec = REGISTRY[domain];

  if (!spec) {
    return res.status(404).json({ error: 'Spec not found' });
  }

  const operations = getOperations(domain);
  const schemaCards = Object.entries(getSchemas(domain)).map(([name, schema]) => ({
    title: name,
    description: schema.description || 'OpenAPI schema',
    prompt: `Show schema ${name} in the ${domain} API`
  }));
  const capabilityCards = getCapabilityCards(domain);

  return res.json({
    domain,
    title: spec.info?.title || '',
    version: spec.info?.version || '',
    operations,
    endpoint_cards: operations.map((op) => ({
      title: `${op.method} ${op.path}`,
      description: op.summary || op.description || 'OpenAPI endpoint',
      prompt: `Explain ${op.method} ${op.path} for the ${domain} API`,
      method: op.method,
      path: op.path,
      summary: op.summary || ''
    })),
    schema_cards: schemaCards,
    schemas: schemaCards,
    capability_cards: capabilityCards,
    capabilities: capabilityCards
  });
});

app.get('/api/openapi-schema', (req, res) => {
  const domain = String(req.query.domain || '');
  const name = String(req.query.name || '');

  const schema = getSchemas(domain)[name];
  if (!schema) {
    return res.status(404).json({ error: 'Schema not found' });
  }

  return res.json({
    domain,
    name,
    raw_schema: schema
  });
});

app.get('/api/openapi-endpoint', (req, res) => {
  const domain = String(req.query.domain || '');
  const routePath = String(req.query.path || '');
  const method = String(req.query.method || '').toUpperCase();

  const op = getOperationByMethodAndPath(domain, method, routePath);
  if (!op) {
    return res.status(404).json({ error: 'Endpoint not found' });
  }

  const detail = extractEndpointDetail(domain, method, routePath);

  return res.json({
    domain,
    method,
    path: routePath,
    summary: op.summary || '',
    description: op.description || '',
    parameters: op.parameters || [],
    requestBody: op.requestBody || null,
    responses: op.responses || {},
    endpoint_parameters: detail?.endpoint_parameters || [],
    related_schema_cards: detail?.related_schema_cards || [],
    request_example: detail?.request_example ?? null,
    response_example: detail?.response_example ?? null,
    raw_openapi: detail?.raw_openapi || op
  });
});

/* -------------------------
 * CHAT
 * ------------------------- */

app.post('/api/chat', async (req, res) => {
  const start = now();
  const id = requestId();

  try {
    const message = String(req.body.message || '');
    const domain = detectDomain(message);
    const lower = message.toLowerCase();
    const endpointPrompt = normalizeEndpointPrompt(message);

    if (!domain) {
      const answer = 'Please specify a domain like Appointment or Multi-Point Inspection.';
      return res.json({
        request_id: id,
        answer,
        sections: toSections(answer),
        source: 'fallback',
        fallback_used: true,
        latency_ms: now() - start
      });
    }

    if (
      lower.includes('rich description') ||
      lower.includes('describe the api') ||
      lower.includes('description of the') ||
      lower.includes('what is the') ||
      lower.includes('overview')
    ) {
      const answer = getOverview(domain);
      return res.json({
        request_id: id,
        answer,
        sections: structuredSectionsFromNumberedText(answer),
        source: 'deterministic',
        fallback_used: false,
        latency_ms: now() - start
      });
    }

    if (
      lower.includes('workflow') ||
      lower.includes('start the inspection') ||
      lower.includes('start inspection') ||
      lower.includes('start the appointment')
    ) {
      const answer = getWorkflowAnswer(domain);
      return res.json({
        request_id: id,
        answer,
        sections: structuredSectionsFromNumberedText(answer),
        source: 'deterministic',
        fallback_used: false,
        latency_ms: now() - start
      });
    }

    if (lower.includes('capabilities') || lower.includes('capability') || lower.includes('customer approval')) {
      const answer = getCapabilityAnswer(domain, message) || 'Capabilities available:';
      return res.json({
        request_id: id,
        answer,
        sections: structuredSectionsFromNumberedText(answer),
        capability_cards: getCapabilityCards(domain),
        source: 'deterministic',
        fallback_used: false,
        latency_ms: now() - start
      });
    }

    if (lower.includes('schema')) {
      const schemas = getSchemas(domain);
      return res.json({
        request_id: id,
        answer: 'Schemas available:',
        sections: toSections('Schemas available:'),
        schema_cards: Object.entries(schemas).map(([name, schema]) => ({
          title: name,
          description: schema.description || 'OpenAPI schema',
          prompt: `Show schema ${name} in the ${domain} API`
        })),
        source: 'openapi',
        fallback_used: false,
        latency_ms: now() - start
      });
    }

    if (endpointPrompt) {
      const parts = endpointPrompt.split(/\s+/);
      const method = parts[0];
      const routePath = parts[1];
      const detail = extractEndpointDetail(domain, method, routePath);

      if (detail) {
        const answer = `1. Direct answer
${method} ${routePath} is a defined endpoint in the ${domain} API.

2. Key endpoint details
${detail.summary || detail.description || 'This endpoint is part of the API surface.'}

3. Why it matters
It provides a direct integration point into the ${domain} workflow.`;

        return res.json({
          request_id: id,
          answer,
          sections: structuredSectionsFromNumberedText(answer),
          endpoint_title: `${method} ${routePath}`,
          endpoint_parameters: detail.endpoint_parameters,
          related_schema_cards: detail.related_schema_cards,
          request_example: detail.request_example,
          response_example: detail.response_example,
          raw_openapi: detail.raw_openapi,
          source: 'openapi',
          fallback_used: false,
          latency_ms: now() - start
        });
      }
    }

    if (lower.includes('endpoint')) {
      return res.json({
        request_id: id,
        answer: 'Available endpoints:',
        sections: toSections('Available endpoints:'),
        endpoint_cards: getOperations(domain).map((op) => ({
          title: `${op.method} ${op.path}`,
          description: op.summary || op.description || 'OpenAPI endpoint',
          prompt: `Explain ${op.method} ${op.path} for the ${domain} API`,
          method: op.method,
          path: op.path,
          summary: op.summary || '',
          raw: op
        })),
        source: 'openapi',
        fallback_used: false,
        latency_ms: now() - start
      });
    }

    if (
      lower.includes('get only') ||
      lower.includes('get operations only') ||
      lower.includes('sub-api') ||
      lower.includes('sub api')
    ) {
      const ops = getOperations(domain).filter((o) => o.method === 'GET');
      return res.json({
        request_id: id,
        answer: 'GET-only sub-API candidate:',
        sections: toSections('GET-only sub-API candidate:'),
        sub_api: ops,
        source: 'openapi',
        fallback_used: false,
        latency_ms: now() - start
      });
    }

    if (!client) {
      const answer = 'OpenAI is not configured on this deployment.';
      return res.json({
        request_id: id,
        answer,
        sections: toSections(answer),
        source: 'fallback',
        fallback_used: true,
        latency_ms: now() - start
      });
    }

    const ai = await openaiCall(
      `Domain: ${domain}
User request: ${message}

Answer clearly in STAR automotive API context.
Be concise and practical.`
    );

    const answer = ai.output_text || 'No response text returned.';
    return res.json({
      request_id: id,
      answer,
      sections: toSections(answer),
      source: 'openai',
      fallback_used: false,
      latency_ms: now() - start
    });
  } catch (e) {
    return res.status(500).json({
      error: e.message || 'Unknown error'
    });
  }
});

/* -------------------------
 * STREAMING
 * ------------------------- */

app.post('/api/chat/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  try {
    const message = String(req.body.message || '');
    const domain = detectDomain(message);
    const lower = message.toLowerCase();

    if (
      domain &&
      (
        lower.includes('workflow') ||
        lower.includes('overview') ||
        lower.includes('rich description')
      )
    ) {
      const answer =
        lower.includes('workflow') ? getWorkflowAnswer(domain) : getOverview(domain);
      res.write(answer || '');
      res.end();
      return;
    }

    if (!client) {
      res.end('OpenAI is not configured.');
      return;
    }

    const stream = await client.responses.stream({
      model: MODEL,
      input: message
    });

    for await (const event of stream) {
      if (event.type === 'response.output_text.delta') {
        res.write(event.delta);
      }
    }

    res.end();
  } catch {
    res.end('Error streaming');
  }
});

/* ------------------------- */

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Demo Mode V3 running on ${PORT}`);
});
