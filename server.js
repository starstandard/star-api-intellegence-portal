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
app.use(express.json());
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

  for (let i = 0; i <= retries; i++) {
    try {
      return await withTimeout(
        client.responses.create({
          model: MODEL,
          input
        }),
        30000,
        'OpenAI answer generation'
      );
    } catch (e) {
      if (i === retries) throw e;
      await new Promise((r) => setTimeout(r, 1000 * (i + 1)));
    }
  }
}

/* -------------------------
 * LOAD OPENAPI
 * ------------------------- */

const ROOT = process.cwd();

const OPENAPI_PATHS = {
  appointment: path.join(ROOT, 'openapi', 'appointment-api.yaml'),
  'multi-point-inspection': path.join(ROOT, 'openapi', 'multi-point-inspection-api.yaml')
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
 * HELPERS
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
  for (const [p, obj] of Object.entries(spec.paths)) {
    for (const m of ['get', 'post', 'put', 'patch', 'delete']) {
      if (obj[m]) {
        ops.push({
          method: m.toUpperCase(),
          path: p,
          summary: obj[m].summary || '',
          description: obj[m].description || '',
          parameters: obj[m].parameters || [],
          requestBody: obj[m].requestBody || null,
          responses: obj[m].responses || {}
        });
      }
    }
  }

  return ops;
}

function normalizeEndpointPrompt(message = '') {
  const match = String(message).match(/(GET|POST|PUT|PATCH|DELETE)\s+\/[A-Za-z0-9/_{}-]+/i);
  return match ? match[0] : null;
}

function getOperationByMethodAndPath(domain, method, routePath) {
  const spec = REGISTRY[domain];
  if (!spec?.paths?.[routePath]) return null;
  return spec.paths[routePath][String(method).toLowerCase()] || null;
}

/* -------------------------
 * DETERMINISTIC RESPONSES
 * ------------------------- */

function deterministicAnswer(message) {
  const m = String(message).toLowerCase();

  if (
    m.includes('inspection workflow') ||
    m.includes('start the inspection workflow') ||
    m.includes('start inspection workflow')
  ) {
    return `1. Direct answer
The inspection workflow begins by creating an Inspection tied to a service visit or repair order.

2. Key details
Technicians then capture findings, attach media, and generate recommendations that advisors review with the customer.

3. Why it matters
This workflow drives transparency, approval decisions, and downstream service execution.`;
  }

  return null;
}

function getBuiltInEndpointDetail(endpoint) {
  if (!endpoint) return null;

  const key = endpoint.toLowerCase();

  if (key.includes('/inspections') && key.includes('/findings')) {
    return {
      answer: `1. Direct answer
GET /inspections/{id}/findings retrieves all findings recorded during an inspection.

2. Key endpoint details
Findings represent vehicle conditions, issues, or observations captured by the technician, often including notes, status, and media.

3. Why it matters
This endpoint supports advisor review, customer communication, and recommendation building.`,
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
    version: 'enterprise-v1'
  });
});

/* -------------------------
 * VERSION + OPENAPI ROUTES
 * ------------------------- */

app.get('/api/version', (_req, res) => {
  res.json({
    version: 'enterprise-v1',
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

  res.json({
    domain,
    title: spec.info?.title || '',
    version: spec.info?.version || '',
    operations
  });
});

app.get('/api/openapi-schema', (req, res) => {
  const domain = String(req.query.domain || '');
  const name = String(req.query.name || '');

  const schemas = getSchemas(domain);
  const schema = schemas[name];

  if (!schema) {
    return res.status(404).json({ error: 'Schema not found' });
  }

  res.json({
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

  res.json({
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
    const audience = String(req.body.audience || 'business');
    const domain = detectDomain(message);

    if (!domain) {
      return res.json({
        request_id: id,
        answer: 'Please specify a domain like Appointment or Multi-Point Inspection.',
        source: 'fallback',
        latency_ms: now() - start
      });
    }

    const det = deterministicAnswer(message);
    if (det) {
      return res.json({
        request_id: id,
        answer: det,
        source: 'deterministic',
        fallback_used: false,
        latency_ms: now() - start
      });
    }

    if (message.toLowerCase().includes('schema')) {
      const schemas = getSchemas(domain);
      return res.json({
        request_id: id,
        answer: 'Schemas available:',
        schema_cards: Object.entries(schemas).map(([name, schema]) => ({
          title: name,
          description: schema.description || ''
        })),
        source: 'openapi',
        fallback_used: false,
        latency_ms: now() - start
      });
    }

    if (message.toLowerCase().includes('endpoint') || normalizeEndpointPrompt(message)) {
      const built = getBuiltInEndpointDetail(message);
      if (built) {
        return res.json({
          request_id: id,
          answer: built.answer,
          endpoint_parameters: built.parameters,
          source: 'deterministic',
          fallback_used: false,
          latency_ms: now() - start
        });
      }

      return res.json({
        request_id: id,
        answer: 'Available endpoints:',
        endpoint_cards: getOperations(domain),
        source: 'openapi',
        fallback_used: false,
        latency_ms: now() - start
      });
    }

    if (message.toLowerCase().includes('get only') || message.toLowerCase().includes('get operations only')) {
      const ops = getOperations(domain).filter((o) => o.method === 'GET');
      return res.json({
        request_id: id,
        answer: 'GET-only sub-API candidate:',
        sub_api: ops,
        source: 'openapi',
        fallback_used: false,
        latency_ms: now() - start
      });
    }

    if (!client) {
      return res.json({
        request_id: id,
        answer: 'OpenAI is not configured on this deployment.',
        source: 'fallback',
        fallback_used: true,
        latency_ms: now() - start
      });
    }

    const ai = await openaiCall(
      `Audience: ${audience}
Domain: ${domain}
User request: ${message}

Explain clearly in STAR automotive API context.`
    );

    res.json({
      request_id: id,
      answer: ai.output_text,
      source: 'openai',
      fallback_used: false,
      latency_ms: now() - start
    });
  } catch (e) {
    res.status(500).json({
      error: e.message || 'Unknown error'
    });
  }
});

/* -------------------------
 * STREAMING
 * ------------------------- */

app.post('/api/chat/stream', async (req, res) => {
  if (!client) {
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.end('OpenAI is not configured.');
    return;
  }

  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  try {
    const message = String(req.body.message || '');

    const det = deterministicAnswer(message);
    if (det) {
      res.write(det);
      res.end();
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
  console.log(`🚀 Enterprise V1 running on ${PORT}`);
});
