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

let rpcId = 1;

function parseMcpResponse(text, contentType = '') {
  const trimmed = String(text || '').trim();
  if (!trimmed) throw new Error('Empty MCP response');
  if (contentType.includes('application/json') && trimmed.startsWith('{')) return JSON.parse(trimmed);
  if (trimmed.startsWith('{')) return JSON.parse(trimmed);

  const dataLines = trimmed
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.replace(/^data:\s*/, '').trim())
    .filter(Boolean);

  if (!dataLines.length) throw new Error(`Invalid MCP response: ${trimmed}`);
  return JSON.parse(dataLines[dataLines.length - 1]);
}

async function callMcp(method, params = {}) {
  const response = await withTimeout(
    fetch(MCP_SERVER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream' },
      body: JSON.stringify({ jsonrpc: '2.0', id: rpcId++, method, params })
    }),
    30000,
    `MCP ${method} request`
  );

  const contentType = response.headers.get('content-type') || '';
  const text = await withTimeout(response.text(), 15000, `MCP ${method} body read`);
  if (!response.ok) throw new Error(`MCP ${method} failed: ${text}`);

  const parsed = parseMcpResponse(text, contentType);
  if (parsed.error) throw new Error(`MCP ${method} error: ${JSON.stringify(parsed.error)}`);
  return parsed.result;
}

async function tool(name, args = {}) {
  return callMcp('tools/call', { name, arguments: args });
}

const cache = new Map();
const CACHE_MS = 5 * 60 * 1000;

function makeCacheKey(name, args = {}) {
  return `${name}:${JSON.stringify(args)}`;
}

async function toolCached(name, args = {}) {
  const key = makeCacheKey(name, args);
  const now = Date.now();
  const cached = cache.get(key);
  if (cached && now - cached.ts < CACHE_MS) return cached.value;
  const value = await tool(name, args);
  cache.set(key, { value, ts: now });
  return value;
}

function detectDomain(message = '') {
  const m = String(message).toLowerCase();
  if (m.includes('multi-point-inspection') || m.includes('multi point inspection') || m.includes('inspection') || m.includes('mpi')) return 'multi-point-inspection';
  if (m.includes('appointment')) return 'appointment';
  return null;
}

function wantsCapabilities(message = '') {
  const m = String(message).toLowerCase();
  return m.includes('capabilities') || m.includes('business capabilities') || m.includes('service advisor') || m.includes('service lane') || m.includes('advisor-facing') || m.includes('daily workflow');
}

function wantsOverview(message = '') {
  const m = String(message).toLowerCase();
  return m.includes('overview') || m.includes('what is') || m.includes('summarize') || m.includes('summary') || m.includes('tell me about') || m.includes('executive overview');
}

function safeParseJson(text) { try { return JSON.parse(text); } catch { return null; } }
function extractOperations(result) { return result?.operations || result?.structuredContent?.operations || []; }
function extractSchemas(result) { return result?.schemas || result?.structuredContent?.schemas || []; }
function extractCapabilityIdentifier(op) { return op?.operationId || op?.identifier || ''; }

function extractSections(answer) {
  const text = String(answer || '').trim();
  if (!text) return [];
  const lines = text.split('\n');
  const sections = [];
  let current = null;
  const isHeader = (line) => /^\s*\d+\.\s+/.test(line) || /^\s*[A-Z][A-Za-z0-9 /&()-]{2,}:\s*$/.test(line);

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { if (current) current.body.push(''); continue; }
    if (isHeader(line)) {
      if (current) sections.push(current);
      current = { title: line.replace(/:\s*$/, ''), body: [] };
    } else {
      if (!current) current = { title: 'Overview', body: [] };
      current.body.push(line);
    }
  }
  if (current) sections.push(current);
  return sections.map((s) => ({ title: s.title, body: s.body.join('\n').trim() })).filter((s) => s.title || s.body);
}

function deriveEndpointCards(result, domain) {
  return extractOperations(result).slice(0, 8).map((op) => ({
    title: `${op.method || 'GET'} ${op.path || ''}`.trim(),
    description: op.summary || op.description || 'API endpoint',
    prompt: `Explain ${(op.method || 'GET')} ${op.path || ''} for the ${domain || 'current'} API`
  }));
}

function deriveSchemaCards(result, domain) {
  return extractSchemas(result).slice(0, 8).map((s) => ({
    title: s.name,
    description: s.description || 'View schema details',
    prompt: `Show schema ${s.name} in the ${domain || 'current'} API`
  }));
}

function defaultCapabilityCards(domain) {
  return [
    { title: 'Start inspection workflow', description: 'Begin or launch the inspection process tied to service intake.', prompt: `Explain how to start the inspection workflow in the ${domain} API for a dealership service advisor` },
    { title: 'Track inspection progress', description: 'Monitor inspection state and advisor-facing workflow progress.', prompt: `Explain how to track inspection progress in the ${domain} API for a dealership service advisor` },
    { title: 'Review findings and media', description: 'Understand technician findings, notes, and media in the service lane.', prompt: `Explain how to review findings and media in the ${domain} API for a dealership service advisor` },
    { title: 'Prepare recommendations', description: 'Build service recommendations and communicate next steps.', prompt: `Explain how to prepare recommendations in the ${domain} API for a dealership service advisor` },
    { title: 'Capture customer approval', description: 'Handle approval workflow and advisor-to-customer communication.', prompt: `Explain how customer approval works in the ${domain} API for a dealership service advisor` },
    { title: 'Close and publish results', description: 'Finish the workflow and share results with internal teams or customers.', prompt: `Explain how to close and publish results in the ${domain} API for a dealership service advisor` }
  ];
}

function operationBasedCapabilityCards(domain, opsResult) {
  return extractOperations(opsResult).slice(0, 6).map((op) => {
    const identifier = extractCapabilityIdentifier(op);
    const title = op.summary || identifier || `${op.method || 'GET'} ${op.path || ''}`.trim() || 'Capability';
    const description = op.description || `${op.method || 'GET'} ${op.path || ''}`.trim() || 'Business capability';
    return {
      title, description, identifier,
      prompt: identifier ? `Explain business use of ${identifier} for a dealership service advisor in the ${domain} API` : `Explain ${title} for a dealership service advisor in the ${domain} API`
    };
  });
}

function defaultWorkflowMap(domain) {
  if (domain === 'appointment') {
    return [
      { step: 'Schedule appointment', detail: 'Customer booking and advisor intake', prompt: 'Explain the scheduling flow of the Appointment API' },
      { step: 'Confirm service needs', detail: 'Advisor validates request and timing', prompt: 'Explain advisor validation workflow in the Appointment API' },
      { step: 'Prepare service lane', detail: 'Appointment data supports dealer operations', prompt: 'Explain how the Appointment API supports service-lane preparation' }
    ];
  }
  return [
    { step: 'Start inspection', detail: 'Launch MPI workflow tied to intake or RO context', prompt: 'Explain how to start the inspection workflow in the multi-point-inspection API for a dealership service advisor' },
    { step: 'Capture findings', detail: 'Technician records results, notes, and media', prompt: 'Explain how findings are captured in the multi-point-inspection API' },
    { step: 'Build recommendations', detail: 'Advisor prepares customer-facing repair guidance', prompt: 'Explain how recommendations are built in the multi-point-inspection API' },
    { step: 'Get approval', detail: 'Customer decisions are captured and tracked', prompt: 'Explain customer approval flow in the multi-point-inspection API' },
    { step: 'Move to execution', detail: 'Approved work moves into repair-order workflow', prompt: 'Explain how approved work moves into execution from the multi-point-inspection API' }
  ];
}

async function buildCapabilities(domain) {
  let ops;
  try {
    ops = await toolCached('listOperations', { domain_name: domain, limit: 12 });
  } catch {
    return {
      answer: `The ${domain} API is responding slowly right now, so I could not build the full business capability map from live operations. You can still continue with these starter capability paths.`,
      sections: extractSections(`1. Direct answer\nThe ${domain} API is responding slowly right now, so I could not build the full business capability map from live operations.\n\n2. Useful next steps\nUse one of the capability cards below to continue exploring advisor workflow guidance.`),
      capability_cards: defaultCapabilityCards(domain),
      endpoint_cards: [],
      schema_cards: [],
      workflow_map: defaultWorkflowMap(domain)
    };
  }

  try {
    const response = await withTimeout(client.responses.create({
      model: MODEL,
      input: `You are a STAR automotive retail strategist.
Return ONLY valid JSON in this shape:
{
  "answer": "string",
  "capability_cards": [{"title":"string","description":"string","identifier":"string","prompt":"string"}],
  "workflow_map": [{"step":"string","detail":"string","prompt":"string"}]
}

Group these operations into dealership-friendly business capabilities for a service advisor.
Use real identifiers when available.
API domain: ${domain}

DATA:
${JSON.stringify(ops)}`
    }), 15000, 'OpenAI capability grouping');

    const parsed = safeParseJson(response.output_text);
    if (!parsed || !Array.isArray(parsed.capability_cards)) throw new Error('Invalid capability JSON');

    return {
      answer: parsed.answer || `Business capabilities for the ${domain} API are shown below.`,
      sections: extractSections(parsed.answer || `Overview\nBusiness capabilities for the ${domain} API are shown below.`),
      capability_cards: parsed.capability_cards,
      endpoint_cards: deriveEndpointCards(ops, domain),
      schema_cards: [],
      workflow_map: Array.isArray(parsed.workflow_map) ? parsed.workflow_map : defaultWorkflowMap(domain)
    };
  } catch {
    return {
      answer: `I found relevant operations for the ${domain} API. Select one of the business capabilities below to continue.`,
      sections: extractSections(`Overview\nI found relevant operations for the ${domain} API. Select one of the business capabilities below to continue.`),
      capability_cards: operationBasedCapabilityCards(domain, ops),
      endpoint_cards: deriveEndpointCards(ops, domain),
      schema_cards: [],
      workflow_map: defaultWorkflowMap(domain)
    };
  }
}

async function buildOverview(domain, message) {
  const [overview, ops, schemas] = await Promise.all([
    toolCached('getApiOverview', { domain_name: domain }),
    toolCached('listOperations', { domain_name: domain, limit: 24 }).catch(() => ({ operations: [] })),
    toolCached('listSchemas', { domain_name: domain, limit: 24 }).catch(() => ({ schemas: [] }))
  ]);

  const response = await withTimeout(client.responses.create({
    model: MODEL,
    input: `You are a STAR automotive retail API architect.
The appointment API means dealer vehicle service appointments, not medical scheduling.
The multi-point-inspection API means vehicle inspection workflow.

User request:
${message}

Overview:
${JSON.stringify(overview)}

Operations:
${JSON.stringify(ops)}

Schemas:
${JSON.stringify(schemas)}

Write a structured answer with:
1. What it is
2. Core concepts
3. Main resources
4. Typical workflow
5. What to explore next`
  }), 15000, 'OpenAI overview generation');

  return {
    answer: response.output_text,
    sections: extractSections(response.output_text),
    capability_cards: [],
    endpoint_cards: deriveEndpointCards(ops, domain),
    schema_cards: deriveSchemaCards(schemas, domain),
    workflow_map: defaultWorkflowMap(domain)
  };
}

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
  res.json({ name: 'star-ai-intelligence-portal', status: 'ok', health: '/health', chat: '/api/chat' });
});

app.post('/api/chat', async (req, res) => {
  try {
    if (!OPENAI_API_KEY) return res.status(500).json({ error: 'Missing OPENAI_API_KEY' });
    if (!MCP_SERVER_URL) return res.status(500).json({ error: 'Missing MCP_SERVER_URL' });

    const message = req.body?.message;
    if (!message || typeof message !== 'string') return res.status(400).json({ error: 'A string "message" is required.' });

    const domain = detectDomain(message);

    if (domain && wantsCapabilities(message)) {
      const result = await buildCapabilities(domain);
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
      const result = await buildOverview(domain, message);
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

    const result = await toolCached('listDomains', {});
    const answer = await withTimeout(client.responses.create({
      model: MODEL,
      input: `User:\n${message}\n\nData:\n${JSON.stringify(result)}\n\nAnswer clearly and concisely in the context of STAR automotive APIs.`
    }), 12000, 'OpenAI answer generation');

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
    if (msg.includes('Too Many Requests')) return res.status(429).json({ error: 'The service is being rate limited right now. Please wait a few seconds and try again.' });
    if (msg.includes('timed out')) return res.status(504).json({ error: msg });
    return res.status(500).json({ error: msg });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`STAR V5 running on ${PORT}`);
});
