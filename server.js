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

if (!OPENAI_API_KEY) {
  console.error('Missing OPENAI_API_KEY');
}
if (!MCP_SERVER_URL) {
  console.error('Missing MCP_SERVER_URL');
}

const client = new OpenAI({ apiKey: OPENAI_API_KEY });
const app = express();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

/* ---------- TIMEOUT HELPER ---------- */

async function withTimeout(promise, ms, label = 'Operation') {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`${label} timed out after ${ms}ms`));
    }, ms);
  });

  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

/* ---------- MCP ---------- */

let id = 1;

function parseMcpResponse(text, contentType = '') {
  const trimmed = String(text || '').trim();

  if (!trimmed) {
    throw new Error('Empty MCP response');
  }

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
        id: id++,
        method,
        params
      })
    }),
    30000,
    `MCP ${method} request`
  );

  const contentType = response.headers.get('content-type') || '';
  const text = await withTimeout(
    response.text(),
    15000,
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

/* ---------- SIMPLE CACHE ---------- */

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

/* ---------- HELPERS ---------- */

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

function safeParseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function extractOperations(opsResult) {
  return opsResult?.operations || opsResult?.structuredContent?.operations || [];
}

function extractCapabilityIdentifier(op) {
  return op?.operationId || op?.identifier || '';
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
    .map((s) => ({
      title: s.title,
      body: s.body.join('\n').trim()
    }))
    .filter((s) => s.title || s.body);
}

function fallbackCapabilityCards(domain) {
  return [
    {
      title: 'Start inspection workflow',
      description: 'Begin or launch the inspection process tied to service intake.',
      identifier: '',
      prompt: `Explain how to start the inspection workflow in the ${domain} API for a dealership service advisor`
    },
    {
      title: 'Track inspection progress',
      description: 'Monitor inspection state and advisor-facing workflow progress.',
      identifier: '',
      prompt: `Explain how to track inspection progress in the ${domain} API for a dealership service advisor`
    },
    {
      title: 'Review findings and media',
      description: 'Understand technician findings, notes, and media in the service lane.',
      identifier: '',
      prompt: `Explain how to review findings and media in the ${domain} API for a dealership service advisor`
    },
    {
      title: 'Prepare recommendations',
      description: 'Build service recommendations and communicate next steps.',
      identifier: '',
      prompt: `Explain how to prepare recommendations in the ${domain} API for a dealership service advisor`
    },
    {
      title: 'Capture customer approval',
      description: 'Handle approval workflow and advisor-to-customer communication.',
      identifier: '',
      prompt: `Explain how customer approval works in the ${domain} API for a dealership service advisor`
    },
    {
      title: 'Close and publish results',
      description: 'Finish the workflow and share results with internal teams or customers.',
      identifier: '',
      prompt: `Explain how to close and publish results in the ${domain} API for a dealership service advisor`
    }
  ];
}

function operationBasedCapabilityCards(domain, opsResult) {
  const operations = extractOperations(opsResult);

  return operations.slice(0, 6).map((op) => {
    const identifier = extractCapabilityIdentifier(op);
    const title =
      op.summary ||
      identifier ||
      `${op.method || 'GET'} ${op.path || ''}`.trim() ||
      'Capability';

    const description =
      op.description ||
      `${op.method || 'GET'} ${op.path || ''}`.trim() ||
      'Business capability';

    return {
      title,
      description,
      identifier,
      prompt: identifier
        ? `Explain business use of ${identifier} for a dealership service advisor in the ${domain} API`
        : `Explain ${title} for a dealership service advisor in the ${domain} API`
    };
  });
}

/* ---------- CAPABILITY MODE ---------- */

async function buildCapabilities(domain) {
  let ops;

  try {
    ops = await toolCached('listOperations', {
      domain_name: domain,
      limit: 10
    });
  } catch (err) {
    console.warn('listOperations fallback:', err.message);

    return {
      answer: `The ${domain} API is responding slowly right now, so I could not build the full business capability map from operations. You can still continue with these starter capability paths.`,
      capability_cards: fallbackCapabilityCards(domain)
    };
  }

  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `Group these operations into business capabilities for a dealership service advisor in automotive retail.

Return ONLY valid JSON in this shape:
{
  "answer": "string",
  "capability_cards": [
    {
      "title": "string",
      "description": "string",
      "identifier": "string",
      "prompt": "string"
    }
  ]
}

Important:
- The "${domain}" API is automotive retail, not medical or generic scheduling.
- Keep the language dealership-friendly.
- Use real operation identifiers when available.
- Create 4 to 8 capability cards.

DATA:
${JSON.stringify(ops)}`
      }),
      15000,
      'OpenAI capability grouping'
    );

    const parsed = safeParseJson(response.output_text);

    if (!parsed || !Array.isArray(parsed.capability_cards)) {
      throw new Error('Model returned invalid capability JSON');
    }

    return parsed;
  } catch (err) {
    console.warn('Capability grouping fallback:', err.message);

    return {
      answer: `I found relevant operations for the ${domain} API. Select one of the business capabilities below to continue.`,
      capability_cards: operationBasedCapabilityCards(domain, ops)
    };
  }
}

/* ---------- ROUTES ---------- */

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
      const nav = await buildCapabilities(domain);

      return res.json({
        answer: nav.answer,
        sections: extractSections(nav.answer),
        capability_cards: nav.capability_cards,
        tool_name: 'business_capability_navigator',
        tool_arguments: { domain_name: domain }
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
      12000,
      'OpenAI answer generation'
    );

    return res.json({
      answer: answer.output_text,
      sections: extractSections(answer.output_text),
      capability_cards: [],
      tool_name: 'listDomains',
      tool_arguments: {}
    });
  } catch (e) {
    console.error(e);

    const msg = e?.message || 'Unknown error';

    if (msg.includes('Too Many Requests')) {
      return res.status(429).json({
        error: 'The service is being rate limited right now. Please wait a few seconds and try again.'
      });
    }

    if (msg.includes('timed out')) {
      return res.status(504).json({
        error: msg
      });
    }

    return res.status(500).json({ error: msg });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`V4.1 running on ${PORT}`);
});
