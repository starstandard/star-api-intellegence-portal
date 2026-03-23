
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
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-5';

if (!OPENAI_API_KEY) console.error('Missing OPENAI_API_KEY');
if (!MCP_SERVER_URL) console.error('Missing MCP_SERVER_URL');

const client = new OpenAI({ apiKey: OPENAI_API_KEY });
const app = express();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

let mcpRequestId = 1;
const toolCallCache = new Map();
const TOOL_CALL_CACHE_MS = 5 * 60 * 1000;

function parseMcp(text) {
  const trimmed = String(text || '').trim();
  if (!trimmed) throw new Error('Empty MCP response');
  if (trimmed.startsWith('{')) return JSON.parse(trimmed);
  const dataLines = trimmed.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.replace(/^data:\s*/, '').trim()).filter(Boolean);
  if (!dataLines.length) throw new Error(`Invalid MCP response: ${trimmed}`);
  return JSON.parse(dataLines[dataLines.length - 1]);
}

async function callMcp(method, params = {}) {
  const response = await fetch(MCP_SERVER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream' },
    body: JSON.stringify({ jsonrpc: '2.0', id: mcpRequestId++, method, params })
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`MCP ${method} failed: ${text}`);
  const parsed = parseMcp(text);
  if (parsed.error) throw new Error(`MCP ${method} error: ${JSON.stringify(parsed.error)}`);
  return parsed.result;
}

async function callTool(name, args = {}) {
  return callMcp('tools/call', { name, arguments: args });
}

function makeToolCacheKey(name, args = {}) {
  return `${name}:${JSON.stringify(args)}`;
}

async function callToolCached(name, args = {}) {
  const key = makeToolCacheKey(name, args);
  const now = Date.now();
  const cached = toolCallCache.get(key);
  if (cached && now - cached.timestamp < TOOL_CALL_CACHE_MS) return cached.value;
  const value = await callTool(name, args);
  toolCallCache.set(key, { value, timestamp: now });
  return value;
}

function extractStructuredContent(toolResult) {
  if (!toolResult) return null;
  return toolResult.structuredContent || toolResult.content || toolResult;
}

function safeJsonParse(text) { try { return JSON.parse(text); } catch { return null; } }

function extractJsonObject(text) {
  const direct = safeJsonParse(text);
  if (direct) return direct;
  const fenced = String(text || '').match(/```json\s*([\s\S]*?)\s*```/i);
  if (fenced?.[1]) {
    const parsed = safeJsonParse(fenced[1]);
    if (parsed) return parsed;
  }
  const firstBrace = String(text || '').indexOf('{');
  const lastBrace = String(text || '').lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    return safeJsonParse(String(text || '').slice(firstBrace, lastBrace + 1));
  }
  return null;
}

function detectDomain(message) {
  const m = String(message || '').toLowerCase();
  if (m.includes('multi-point-inspection') || m.includes('multi point inspection') || m.includes('mpi') || (m.includes('inspection') && !m.includes('appointment'))) return 'multi-point-inspection';
  if (m.includes('appointment')) return 'appointment';
  return null;
}

function isOverview(message) {
  const m = String(message || '').toLowerCase();
  return m.includes('overview') || m.includes('what is') || m.includes('explain') || m.includes('summarize') || m.includes('summary') || m.includes('tell me about');
}

function deriveDomainInsights(operationsResult, schemasResult) {
  const ops = operationsResult?.structuredContent?.operations || operationsResult?.operations || [];
  const schemas = schemasResult?.structuredContent?.schemas || schemasResult?.schemas || [];
  const pathCounts = new Map();
  const exampleEndpoints = [];
  const conceptHints = new Set();
  const schemaNames = [];

  for (const op of ops) {
    const path = op.path || '';
    const method = op.method || op.http_method || 'GET';
    const segments = String(path).split('/').filter(Boolean);
    const top = segments[0] || 'root';
    pathCounts.set(top, (pathCounts.get(top) || 0) + 1);
    if (path && exampleEndpoints.length < 10) {
      exampleEndpoints.push({ title: `${method} ${path}`, method, path, description: op.summary || op.description || 'Representative API endpoint' });
    }
    const hay = `${op.summary || ''} ${op.description || ''} ${path}`.toLowerCase();
    ['manifest','inspection','labor','media','finding','event','part','price','repair'].forEach((t) => {
      if (hay.includes(t)) conceptHints.add(t === 'labor' ? 'LaborOperation' : t.charAt(0).toUpperCase() + t.slice(1));
    });
    if (hay.includes('partyreference') || hay.includes('party reference')) conceptHints.add('PartyReference');
  }

  for (const schema of schemas.slice(0, 80)) {
    const name = schema.name || '';
    if (name) schemaNames.push(name);
    const lower = name.toLowerCase();
    if (lower.includes('partyreference') || lower.includes('party_reference')) conceptHints.add('PartyReference');
    ['manifest','inspection','labor','media','finding','event','part','price','repair'].forEach((t) => {
      if (lower.includes(t)) conceptHints.add(t === 'labor' ? 'LaborOperation' : t.charAt(0).toUpperCase() + t.slice(1));
    });
  }

  return {
    operation_count: ops.length,
    schema_count: schemas.length,
    top_resource_groups: Array.from(pathCounts.entries()).sort((a,b)=>b[1]-a[1]).slice(0,10).map(([name,operation_count])=>({name,operation_count})),
    candidate_core_concepts: Array.from(conceptHints),
    example_endpoints: exampleEndpoints,
    sample_schema_names: schemaNames.slice(0, 20)
  };
}

function buildExploreNext(domain, derived = {}) {
  if (!domain) return [];
  const prompts = [
    `List operations for the ${domain} API`,
    `List schemas for the ${domain} API`,
    `Show example endpoints for the ${domain} API`
  ];
  for (const concept of (derived.candidate_core_concepts || []).slice(0, 3)) {
    prompts.push(`Show me key schemas and operations related to ${concept} in the ${domain} API`);
  }
  return Array.from(new Set(prompts)).slice(0, 6);
}

function schemaCardsFromNames(names, domain) {
  return (names || []).slice(0, 8).map((name) => ({
    title: name,
    description: `View schema details for ${name}`,
    prompt: `Show schema ${name} in the ${domain} API`
  }));
}

function schemaCards(result, domain) {
  const schemas = result?.schemas || result?.structuredContent?.schemas || [];
  return schemas.slice(0, 8).map((s) => ({
    title: s.name,
    description: s.description || 'View schema details',
    prompt: `Show schema ${s.name} in the ${domain || 'current'} API`
  }));
}

function endpointCards(derived = {}, domain) {
  return (derived.example_endpoints || []).slice(0, 8).map((ep) => ({
    title: ep.title,
    description: ep.description,
    prompt: `Explain ${ep.title} for the ${domain} API`
  }));
}

function genericEndpointCards(toolResult, domain) {
  const ops = toolResult?.operations || toolResult?.structuredContent?.operations || [];
  return ops.slice(0, 8).map((op) => ({
    title: `${op.method || op.http_method || 'GET'} ${op.path || ''}`.trim(),
    description: op.summary || op.description || 'API endpoint',
    prompt: `Explain ${(op.method || op.http_method || 'GET')} ${op.path || ''} for the ${domain || 'current'} API`
  }));
}

function executiveDemoPrompts() {
  return [
    'Give me an executive overview of the Appointment API',
    'Give me an executive overview of the Multi-Point Inspection API',
    'Compare the Appointment API and Multi-Point Inspection API',
    'Show me the core workflow of the Multi-Point Inspection API',
    'Explain how the Appointment API fits into the dealer service journey'
  ];
}

function extractSections(answer) {
  const text = String(answer || '').trim();
  if (!text) return [];
  const lines = text.split('\n');
  const sections = [];
  let current = null;
  const isHeader = (line) => /^\s*\d+\.\s+/.test(line) || /^\s*[A-Z][A-Za-z0-9 /&-]{2,}:\s*$/.test(line);
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
  return sections.map((s) => ({ title: s.title, body: s.body.join('\n').trim() })).filter((s) => s.body || s.title);
}

async function buildDomainOverview(domain, message) {
  const overview = await callToolCached('getApiOverview', { domain_name: domain });
  const schemas = await callToolCached('listSchemas', { domain_name: domain });
  const operations = await callToolCached('listOperations', { domain_name: domain, limit: 80 });
  const derived = deriveDomainInsights(operations, schemas);

  const prompt = `
You are an expert STAR automotive retail API architect.
The "appointment" API means AUTOMOTIVE DEALER SERVICE APPOINTMENT, not medical scheduling.
The "multi-point-inspection" API means VEHICLE INSPECTION WORKFLOW in automotive retail service.

User request:
${message}

API:
${domain}

API overview:
${JSON.stringify(extractStructuredContent(overview), null, 2)}

Schemas:
${JSON.stringify(extractStructuredContent(schemas), null, 2)}

Operations:
${JSON.stringify(extractStructuredContent(operations), null, 2)}

Derived insights:
${JSON.stringify(derived, null, 2)}

Write a rich, grounded overview with these sections:
1. What it is
2. Core concepts
3. Main resources
4. Typical workflow
5. Important schemas
6. Representative endpoints
7. What to explore next

Rules:
- Keep everything in automotive retail / dealer service context.
- Do NOT drift into medical or generic appointment language.
- Be concrete and structured.
- Use short paragraphs and bullets where helpful.
- Include example endpoints only if supported by the data.
- Do NOT ask follow-up questions.
- Do NOT say "Would you like me to..."
- End with actionable next-step suggestions.
`;

  const response = await client.responses.create({ model: OPENAI_MODEL, input: prompt });

  return {
    answer: response.output_text,
    sections: extractSections(response.output_text),
    tool_name: 'overview_pipeline',
    tool_arguments: { domain },
    explore_next: buildExploreNext(domain, derived),
    schema_cards: schemaCardsFromNames(derived.sample_schema_names, domain),
    endpoint_cards: endpointCards(derived, domain),
    executive_demo_prompts: executiveDemoPrompts()
  };
}

async function chooseTool(message, domain) {
  const prompt = `
You are choosing the best MCP tool for a STAR automotive retail metadata portal.

Available tools:
- listDomains
- getApiOverview
- listOperations
- listSchemas
- getSchema
- getOperationDetails
- explainOperationForBusiness
- suggestSubApis
- generateConsumerSummary

User message:
${message}

Detected API:
${domain || 'none'}

Return ONLY JSON:
{"tool_name":"string","arguments":{}}

Rules:
- Prefer automotive dealer service interpretation.
- If the user asks about schemas, prefer listSchemas or getSchema.
- If the user asks about operations/endpoints/URLs, prefer listOperations.
- If the user asks for an overview, prefer getApiOverview.
- If API is known, include domain_name.
`;
  const response = await client.responses.create({ model: OPENAI_MODEL, input: prompt });
  const plan = extractJsonObject(response.output_text || '');
  if (!plan?.tool_name || typeof plan.arguments !== 'object') {
    return {
      tool_name: message.toLowerCase().includes('schema') ? 'listSchemas' : 'listDomains',
      arguments: domain ? { domain_name: domain } : {}
    };
  }
  return plan;
}

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    model: OPENAI_MODEL,
    mcp: MCP_SERVER_URL,
    cached_tool_results: toolCallCache.size
  });
});

app.post('/api/chat', async (req, res) => {
  try {
    if (!OPENAI_API_KEY) return res.status(500).json({ error: 'Missing OPENAI_API_KEY' });
    if (!MCP_SERVER_URL) return res.status(500).json({ error: 'Missing MCP_SERVER_URL' });

    const message = req.body?.message;
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'A string "message" is required.' });
    }

    const domain = detectDomain(message);

    if (domain && isOverview(message)) {
      return res.json(await buildDomainOverview(domain, message));
    }

    const plan = await chooseTool(message, domain);
    const toolResult = await callToolCached(plan.tool_name, plan.arguments || {});
    const structured = extractStructuredContent(toolResult);

    const answerPrompt = `
You are a STAR automotive retail API assistant.
Important grounding:
- "appointment" means vehicle service appointment in automotive retail.
- "multi-point-inspection" means vehicle inspection workflow.
- Never drift into medical appointment language.

User request:
${message}

Tool used:
${plan.tool_name}

Tool arguments:
${JSON.stringify(plan.arguments || {}, null, 2)}

Tool result:
${JSON.stringify(structured, null, 2)}

Write a clear answer with:
1. Direct answer
2. Key details
3. Useful next steps

Rules:
- Be grounded in the provided MCP data.
- Do NOT ask vague follow-up questions.
- Do NOT say "Would you like me to..."
- Prefer actionable next steps.
- If relevant, mention schemas or representative endpoints.
`;
    const answer = await client.responses.create({ model: OPENAI_MODEL, input: answerPrompt });
    const cardsDomain = domain || plan.arguments?.domain_name || null;

    return res.json({
      answer: answer.output_text,
      sections: extractSections(answer.output_text),
      tool_name: plan.tool_name,
      tool_arguments: plan.arguments || {},
      explore_next: buildExploreNext(cardsDomain),
      schema_cards: schemaCards(toolResult, cardsDomain),
      endpoint_cards: genericEndpointCards(toolResult, cardsDomain),
      executive_demo_prompts: executiveDemoPrompts()
    });
  } catch (e) {
    console.error(e);
    const msg = e?.message || 'Unknown error';
    if (msg.includes('Too Many Requests')) {
      return res.status(429).json({
        error: 'The MCP server is being rate limited right now. Please wait a few seconds and try again.'
      });
    }
    return res.status(500).json({ error: msg });
  }
});

app.listen(PORT, () => {
  console.log('🚀 Server running on port', PORT);
});
