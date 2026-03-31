import 'dotenv/config';
import express from 'express';
import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import crypto from 'crypto';

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3000;
const MODEL = process.env.OPENAI_MODEL || 'gpt-5';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// -------------------------
// UTILITIES
// -------------------------
const now = () => Date.now();

function requestId() {
  return crypto.randomUUID();
}

async function withTimeout(promise, ms) {
  let timeout;
  const timer = new Promise((_, reject) => {
    timeout = setTimeout(() => reject(new Error('timeout')), ms);
  });
  try {
    return await Promise.race([promise, timer]);
  } finally {
    clearTimeout(timeout);
  }
}

async function openaiCall(input, retries = 2) {
  for (let i = 0; i <= retries; i++) {
    try {
      return await withTimeout(
        client.responses.create({
          model: MODEL,
          input
        }),
        30000
      );
    } catch (e) {
      if (i === retries) throw e;
      await new Promise(r => setTimeout(r, 1000 * (i + 1)));
    }
  }
}

// -------------------------
// LOAD OPENAPI
// -------------------------
const ROOT = process.cwd();

const OPENAPI_PATHS = {
  appointment: path.join(ROOT, 'openapi/appointment-api.yaml'),
  'multi-point-inspection': path.join(ROOT, 'openapi/multi-point-inspection-api.yaml')
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

// -------------------------
// HELPERS
// -------------------------
function detectDomain(msg = '') {
  const m = msg.toLowerCase();
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
    for (const m of ['get','post','put','patch','delete']) {
      if (obj[m]) {
        ops.push({
          method: m.toUpperCase(),
          path: p,
          summary: obj[m].summary || ''
        });
      }
    }
  }
  return ops;
}

// -------------------------
// DETERMINISTIC RESPONSES
// -------------------------
function deterministicAnswer(message) {
  const m = message.toLowerCase();

  if (m.includes('inspection workflow')) {
    return `Inspection workflow starts by creating an inspection tied to a service visit, then capturing findings, media, and recommendations, followed by advisor review and customer approval.`;
  }

  return null;
}

// -------------------------
// API ROUTES
// -------------------------
app.get('/api/version', (_req, res) => {
  res.json({
    version: 'enterprise-v1',
    domains: Object.keys(REGISTRY)
  });
});

app.get('/api/openapi-index', (req, res) => {
  const domain = req.query.domain;
  const spec = REGISTRY[domain];
  if (!spec) return res.status(404).json({ error: 'Spec not found' });

  res.json({ paths: Object.keys(spec.paths || {}) });
});

app.get('/api/openapi-schema', (req, res) => {
  const { domain, name } = req.query;
  const schemas = getSchemas(domain);
  res.json(schemas[name] || {});
});

app.get('/api/openapi-endpoint', (req, res) => {
  const { domain, path: p, method } = req.query;
  const spec = REGISTRY[domain];
  const ep = spec?.paths?.[p]?.[method?.toLowerCase()];
  res.json(ep || {});
});

// -------------------------
// CHAT (MAIN)
// -------------------------
app.post('/api/chat', async (req, res) => {
  const start = now();
  const id = requestId();

  try {
    const message = req.body.message || '';
    const audience = req.body.audience || 'business';
    const domain = detectDomain(message);

    if (!domain) {
      return res.json({ request_id: id, answer: 'Specify domain.' });
    }

    // deterministic first
    const det = deterministicAnswer(message);
    if (det) {
      return res.json({
        request_id: id,
        answer: det,
        source: 'deterministic',
        latency_ms: now() - start
      });
    }

    // schema
    if (message.toLowerCase().includes('schema')) {
      const schemas = getSchemas(domain);
      return res.json({
        request_id: id,
        schema_cards: Object.keys(schemas).map(s => ({ title: s })),
        latency_ms: now() - start
      });
    }

    // endpoints
    if (message.toLowerCase().includes('endpoint')) {
      return res.json({
        request_id: id,
        endpoint_cards: getOperations(domain),
        latency_ms: now() - start
      });
    }

    // sub-api builder
    if (message.toLowerCase().includes('get only')) {
      const ops = getOperations(domain).filter(o => o.method === 'GET');
      return res.json({
        request_id: id,
        sub_api: ops,
        latency_ms: now() - start
      });
    }

    // AI fallback
    const ai = await openaiCall(
      `Audience: ${audience}\nDomain: ${domain}\nUser: ${message}`
    );

    res.json({
      request_id: id,
      answer: ai.output_text,
      source: 'openai',
      latency_ms: now() - start
    });

  } catch (e) {
    res.status(500).json({
      error: e.message
    });
  }
});

// -------------------------
// STREAMING
// -------------------------
app.post('/api/chat/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/plain');

  try {
    const message = req.body.message;

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

// -------------------------
app.get('/', (_req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'star-ai-intelligence-portal'
  });
});

app.get('/health', (_req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'star-ai-intelligence-portal',
    version: 'enterprise-v1'
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Enterprise V1 running on ${PORT}`);
});
