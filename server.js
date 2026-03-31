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
      current = {
        title: match[2],
        body: []
      };
    } else {
      if (!current) {
        current = { title: 'Direct answer', body: [] };
      }
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

async function openaiCall(input, retries = 2) {
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
        30000,
        'OpenAI answer generation'
      );
    } catch (error) {
      if (i === retries) throw error;
      await new Promise((resolve) => setTimeout(resolve, 1000 * (i + 1)));
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

/* -------------------------
 * DETERMINISTIC CONTENT
 * ------------------------- */

function deterministicWorkflowAnswer(domain, audience = 'business') {
  if (domain !== 'multi-point-inspection') return null;

  if (audience === 'architecture') {
    return `1. Direct answer
The inspection workflow begins when an Inspection is established as the core domain object tied to a service event.

2. Key details
That inspection becomes the anchor for findings, media, recommendations, and approval-related outcomes across the workflow.

3. Why it matters
It creates a stable boundary between upstream service intake and downstream execution decisions.`;
  }

  if (audience === 'technical') {
    return `1. Direct answer
The inspection workflow starts when the system creates an Inspection resource for a service visit or repair context.

2. Key details
Technicians then populate findings, attach media, and generate recommendation-related data that advisors can review.

3. Why it matters
This is the main lifecycle path that drives retrieval, update, and customer-decision flows in the API.`;
  }

  return `1. Direct answer
The inspection workflow begins by creating an Inspection tied to a vehicle service visit or repair order.

2. Key details
Technicians capture findings and media, then advisors review results and recommendations with the customer.

3. Why it matters
This workflow supports transparency, trust, and downstream service execution.`;
}

function deterministicOverview(domain, audience = 'business') {
  if (domain === 'multi-point-inspection') {
    if (audience === 'architecture') {
      return `1. Direct answer
The Multi-Point Inspection API is the inspection and recommendation domain for dealership service workflows.

2. Core responsibilities
It manages inspections, findings, media, recommendations, and approval-related outcomes while connecting upstream service context to downstream execution decisions.

3. Why it matters
It transforms inspection data into actionable, customer-facing service intelligence.`;
    }

    if (audience === 'technical') {
      return `1. Direct answer
The Multi-Point Inspection API models inspections, findings, media, recommendations, and customer-decision workflow.

2. Core responsibilities
It provides the resource structures and operations needed to create, retrieve, and reason about inspection outcomes.

3. Why it matters
It is the central technical contract for MPI-driven service-lane workflows.`;
    }

    return `1. Direct answer
The Multi-Point Inspection API supports structured vehicle inspections in the dealership service lane.

2. Core responsibilities
It manages inspection creation, findings, technician observations, media, recommendations, and approval-related workflow.

3. Why it matters
It helps teams explain vehicle condition clearly and move from inspection to approved service work.`;
  }

  if (domain === 'appointment') {
    if (audience === 'architecture') {
      return `1. Direct answer
The Appointment API is the scheduling and intake boundary for dealership service operations.

2. Core responsibilities
It manages appointment creation, requested service context, timing, and readiness for downstream service workflows.

3. Why it matters
It provides the upstream coordination layer that later domains depend on.`;
    }

    if (audience === 'technical') {
      return `1. Direct answer
The Appointment API models service scheduling and intake-related appointment data.

2. Core responsibilities
It represents appointments, requested services, timing context, and retrieval flows for service-lane preparation.

3. Why it matters
It is the primary technical contract for appointment-driven service workflows.`;
    }

    return `1. Direct answer
The Appointment API helps dealerships schedule and manage service appointments.

2. Core responsibilities
It captures booking context, requested services, and timing details needed before the visit begins.

3. Why it matters
It helps prepare the service lane and align customer expectations with dealership operations.`;
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
Those findings typically represent technician observations, condition results, notes, and related media or evidence.

3. Why it matters
This is a core read path for advisor review, customer communication, and recommendation building.`,
      parameters: [
        { name: 'id', in: 'path', required: true, type: 'string' }
      ]
    };
  }

  return null;
}

/* -------------------------
 * HEALTH + UI
 * ------------------------- */

app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/health', (_req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'star-ai-intelligence-portal',
    version: 'enterprise-v2'
  });
});

/* -------------------------
 * VERSION + OPENAPI ROUTES
 * ------------------------- */

app.get('/api/version', (_req, res) => {
  res.json({
    version: 'enterprise-v2',
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

  return res.json({
    domain,
    title: spec.info?.title || '',
    version: spec.info?.version || '',
    operations: getOperations(domain)
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

  return res.json({
    domain,
    method,
    path: routePath,
    summary: op.summary || '',
    description: op.description || '',
    parameters: op.parameters || [],
    requestBody: op.requestBody || null,
    responses: op.responses || {}
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
    const audience = String(req.body.audience || 'business').toLowerCase();
    const domain = detectDomain(message);
    const lower = message.toLowerCase();

    if (!domain) {
      return res.json({
        request_id: id,
        answer: 'Please specify a domain like Appointment or Multi-Point Inspection.',
        sections: toSections('Please specify a domain like Appointment or Multi-Point Inspection.'),
        source: 'fallback',
        fallback_used: true,
        latency_ms: now() - start
      });
    }

    if (
      lower.includes('inspection workflow') ||
      lower.includes('start the inspection workflow') ||
      lower.includes('start inspection workflow')
    ) {
      const answer = deterministicWorkflowAnswer(domain, audience);
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
      lower.includes('rich description') ||
      lower.includes('describe the api') ||
      lower.includes('description of the') ||
      lower.includes('what is the') ||
      lower.includes('overview')
    ) {
      const answer = deterministicOverview(domain, audience);
      if (answer) {
        return res.json({
          request_id: id,
          answer,
          sections: structuredSectionsFromNumberedText(answer),
          source: 'deterministic',
          fallback_used: false,
          latency_ms: now() - start
        });
      }
    }

    if (lower.includes('schema')) {
      const schemas = getSchemas(domain);
      return res.json({
        request_id: id,
        answer: 'Schemas available:',
        sections: toSections('Schemas available:'),
        schema_cards: Object.entries(schemas).map(([name, schema]) => ({
          title: name,
          description: schema.description || 'OpenAPI schema'
        })),
        source: 'openapi',
        fallback_used: false,
        latency_ms: now() - start
      });
    }

    if (lower.includes('endpoint') || normalizeEndpointPrompt(message)) {
      const built = getBuiltInEndpointDetail(message);
      if (built) {
        return res.json({
          request_id: id,
          answer: built.answer,
          sections: structuredSectionsFromNumberedText(built.answer),
          endpoint_parameters: built.parameters,
          source: 'deterministic',
          fallback_used: false,
          latency_ms: now() - start
        });
      }

      return res.json({
        request_id: id,
        answer: 'Available endpoints:',
        sections: toSections('Available endpoints:'),
        endpoint_cards: getOperations(domain),
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
      `Audience: ${audience}
Domain: ${domain}
User request: ${message}

Explain clearly in STAR automotive API context.
Return a concise but structured response.`
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

    if (
      domain &&
      (
        message.toLowerCase().includes('inspection workflow') ||
        message.toLowerCase().includes('start the inspection workflow')
      )
    ) {
      res.write(deterministicWorkflowAnswer(domain, 'business') || '');
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
  console.log(`🚀 Enterprise V2 running on ${PORT}`);
});
