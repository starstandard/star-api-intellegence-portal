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

let mcpRequestId = 1;

/* ---------------- MCP HELPERS ---------------- */

function parseMcp(text) {
  const trimmed = String(text || '').trim();

  if (!trimmed) {
    throw new Error('Empty MCP response');
  }

  if (trimmed.startsWith('{')) {
    return JSON.parse(trimmed);
  }

  const dataLines = trimmed
    .split('\n')
    .filter((l) => l.startsWith('data:'))
    .map((l) => l.replace(/^data:\s*/, '').trim())
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

  const response = await fetch(MCP_SERVER_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream'
    },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: mcpRequestId++,
      method,
      params
    })
  });

  const text = await response.text();

  if (!response.ok) {
    throw new Error(`MCP ${method} failed: ${text}`);
  }

  const parsed = parseMcp(text);

  if (parsed.error) {
    throw new Error(`MCP ${method} error: ${JSON.stringify(parsed.error)}`);
  }

  return parsed.result;
}

async function callTool(name, args = {}) {
  return callMcp('tools/call', {
    name,
    arguments: args
  });
}

/* ---------------- TOOL CALL CACHE ---------------- */

const toolCallCache = new Map();
const TOOL_CALL_CACHE_MS = 5 * 60 * 1000;

function makeToolCacheKey(name, args = {}) {
  return `${name}:${JSON.stringify(args)}`;
}

async function callToolCached(name, args = {}) {
  const key = makeToolCacheKey(name, args);
  const now = Date.now();
  const cached = toolCallCache.get(key);

  if (cached && now - cached.timestamp < TOOL_CALL_CACHE_MS) {
    console.log('⚡ Using cached MCP result for', key);
    return cached.value;
  }

  const value = await callTool(name, args);

  toolCallCache.set(key, {
    value,
    timestamp: now
  });

  return value;
}

/* ---------------- GENERAL HELPERS ---------------- */

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

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
    const parsed = safeJsonParse(String(text || '').slice(firstBrace, lastBrace + 1));
    if (parsed) return parsed;
  }

  return null;
}

function extractStructuredContent(toolResult) {
  if (!toolResult) return null;
  if (toolResult.structuredContent) return toolResult.structuredContent;
  if (toolResult.content) return toolResult.content;
  return toolResult;
}

/* ---------------- INTENT DETECTION ---------------- */

function detectDomain(message) {
  const m = String(message || '').toLowerCase();

  if (
    m.includes('multi-point-inspection') ||
    m.includes('multi point inspection') ||
    m.includes('mpi') ||
    (m.includes('inspection') && !m.includes('appointment'))
  ) {
    return 'multi-point-inspection';
  }

  if (m.includes('appointment')) {
    return 'appointment';
  }

  return null;
}

function isOverview(message) {
  const m = String(message || '').toLowerCase();
  return (
    m.includes('overview') ||
    m.includes('what is') ||
    m.includes('explain') ||
    m.includes('summarize') ||
    m.includes('summary') ||
    m.includes('tell me about')
  );
}

function wantsOperations(message) {
  const m = String(message || '').toLowerCase();
  return m.includes('operation') || m.includes('endpoint') || m.includes('url') || m.includes('path');
}

function wantsSchemaDetails(message) {
  return String(message || '').toLowerCase().includes('schema');
}

/* ---------------- DERIVED DOMAIN INSIGHTS ---------------- */

function deriveDomainInsights(operationsResult, schemasResult) {
  const ops =
    operationsResult?.structuredContent?.operations ||
    operationsResult?.operations ||
    [];

  const schemas =
    schemasResult?.structuredContent?.schemas ||
    schemasResult?.schemas ||
    [];

  const pathCounts = new Map();
  const tagCounts = new Map();
  const exampleEndpoints = [];
  const conceptHints = new Set();
  const schemaNames = [];

  for (const op of ops) {
    const path = op.path || '';
    const method = op.method || op.http_method || 'GET';
    const segments = String(path).split('/').filter(Boolean);
    const top = segments[0] || 'root';

    pathCounts.set(top, (pathCounts.get(top) || 0) + 1);

    if (path && exampleEndpoints.length < 12) {
      exampleEndpoints.push(`${method} ${path}`);
    }

    const tags = Array.isArray(op.tags) ? op.tags : [];
    for (const tag of tags) {
      const key = String(tag);
      tagCounts.set(key, (tagCounts.get(key) || 0) + 1);
    }

    const hay = `${op.summary || ''} ${op.description || ''} ${path}`.toLowerCase();

    if (hay.includes('manifest')) conceptHints.add('Manifest');
    if (hay.includes('inspection')) conceptHints.add('Inspection');
    if (hay.includes('labor')) conceptHints.add('LaborOperation');
    if (hay.includes('media')) conceptHints.add('Media');
    if (hay.includes('finding')) conceptHints.add('Finding');
    if (hay.includes('event')) conceptHints.add('Event');
    if (hay.includes('part')) conceptHints.add('Part');
    if (hay.includes('price')) conceptHints.add('Pricing');
    if (hay.includes('partyreference') || hay.includes('party reference')) conceptHints.add('PartyReference');
    if (hay.includes('repair')) conceptHints.add('Repair');
  }

  for (const schema of schemas.slice(0, 80)) {
    const name = schema.name || '';
    schemaNames.push(name);

    const lower = name.toLowerCase();
    if (lower.includes('manifest')) conceptHints.add('Manifest');
    if (lower.includes('inspection')) conceptHints.add('Inspection');
    if (lower.includes('labor')) conceptHints.add('LaborOperation');
    if (lower.includes('media')) conceptHints.add('Media');
    if (lower.includes('finding')) conceptHints.add('Finding');
    if (lower.includes('event')) conceptHints.add('Event');
    if (lower.includes('part')) conceptHints.add('Part');
    if (lower.includes('price')) conceptHints.add('Pricing');
    if (lower.includes('partyreference') || lower.includes('party_reference')) conceptHints.add('PartyReference');
    if (lower.includes('repair')) conceptHints.add('Repair');
  }

  const topResourceGroups = Array.from(pathCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
    .map(([name, operation_count]) => ({ name, operation_count }));

  const topTags = Array.from(tagCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
    .map(([name, operation_count]) => ({ name, operation_count }));

  return {
    operation_count: ops.length,
    schema_count: schemas.length,
    top_resource_groups: topResourceGroups,
    top_tags: topTags,
    candidate_core_concepts: Array.from(conceptHints),
    example_endpoints: exampleEndpoints,
    sample_schema_names: schemaNames.slice(0, 24)
  };
}

/* ---------------- UI HELPERS ---------------- */

function exploreNext(domain, derived = {}) {
  if (!domain) return [];

  const prompts = [
    `List operations for the ${domain} domain`,
    `List schemas for the ${domain} domain`,
    `Show example endpoints for the ${domain} domain`
  ];

  for (const concept of (derived.candidate_core_concepts || []).slice(0, 3)) {
    prompts.push(`Show me key schemas and operations related to ${concept} in the ${domain} domain`);
  }

  return Array.from(new Set(prompts)).slice(0, 6);
}

function schemaCards(result, domain) {
  const schemas =
    result?.schemas ||
    result?.structuredContent?.schemas ||
    [];

  return schemas.slice(0, 8).map((s) => ({
    title: s.name,
    description: s.description || 'View schema details',
    prompt: `Show schema ${s.name} in the ${domain} domain`
  }));
}

function schemaCardsFromNames(schemaNames, domain) {
  return (schemaNames || []).slice(0, 8).map((name) => ({
    title: name,
    description: `View schema details for ${name}`,
    prompt: `Show schema ${name} in the ${domain} domain`
  }));
}

/* ---------------- OVERVIEW PIPELINE ---------------- */

async function buildDomainOverview(domain, message) {
  const overview = await callToolCached('getApiOverview', { domain_name: domain });
  const schemas = await callToolCached('listSchemas', { domain_name: domain });

  // Only add operations when the user seems to want that detail.
  const operations = wantsOperations(message)
    ? await callToolCached('listOperations', { domain_name: domain, limit: 100 })
    : null;

  const derived = deriveDomainInsights(
    operations || { operations: [] },
    schemas
  );

  const prompt = `
You are an expert STAR automotive retail API architect.

You are explaining a STAR automotive retail domain.
The "appointment" domain means AUTOMOTIVE DEALER SERVICE APPOINTMENT, not medical scheduling.
The "multi-point-inspection" domain means VEHICLE INSPECTION WORKFLOW in automotive retail service.

User request:
${message}

Domain:
${domain}

Domain overview:
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
- Do NOT drift into medical, calendar, or generic appointment language.
- Be concrete and structured.
- Use short paragraphs and bullets where helpful.
- Mention likely central objects if supported by the data.
- Include example endpoints only if supported by the data.
- Do NOT ask follow-up questions.
- Do NOT say "Would you like me to..."
- End with actionable next-step suggestions.
`;

  const response = await client.responses.create({
    model: OPENAI_MODEL,
    input: prompt
  });

  return {
    answer: response.output_text,
    tool_name: 'overview_pipeline',
    tool_arguments: { domain },
    explore_next: exploreNext(domain, derived),
    schema_cards: schemaCardsFromNames(derived.sample_schema_names, domain)
  };
}

/* ---------------- GENERIC TOOL PLANNER ---------------- */

async function chooseTool(message, domain) {
  const prompt = `
You are choosing the best MCP tool for a STAR automotive retail metadata portal.

Available tools:
- listDomains: list available automotive API domains
- getApiOverview: high-level summary of one automotive domain
- listOperations: list operations/endpoints for one domain
- listSchemas: list schema names for one domain
- getSchema: get one schema by exact schema name
- getOperationDetails: get one operation by identifier
- explainOperationForBusiness: explain one operation in plain language
- suggestSubApis: suggest smaller sub-APIs
- generateConsumerSummary: generate concise domain summary

User message:
${message}

Detected domain:
${domain || 'none'}

Return ONLY JSON:
{
  "tool_name": "string",
  "arguments": {}
}

Rules:
- Prefer automotive dealer service interpretation.
- If the user asks about schemas, prefer listSchemas or getSchema.
- If the user asks about operations/endpoints/URLs, prefer listOperations.
- If the user asks for an overview, prefer getApiOverview.
- If domain is known, include domain_name.
- If schema name is not exact, prefer listSchemas first.
`;

  const response = await client.responses.create({
    model: OPENAI_MODEL,
    input: prompt
  });

  const plan = extractJsonObject(response.output_text || '');

  if (!plan?.tool_name || typeof plan.arguments !== 'object') {
    if (wantsSchemaDetails(message)) {
      return {
        tool_name: 'listSchemas',
        arguments: domain ? { domain_name: domain } : {}
      };
    }

    return {
      tool_name: 'listDomains',
      arguments: {}
    };
  }

  return plan;
}

/* ---------------- ROUTES ---------------- */

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

    /* -------- RICH OVERVIEW MODE -------- */
    if (domain && isOverview(message)) {
      const result = await buildDomainOverview(domain, message);
      return res.json(result);
    }

    /* -------- GENERIC MCP ORCHESTRATION -------- */

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

    const answer = await client.responses.create({
      model: OPENAI_MODEL,
      input: answerPrompt
    });

    return res.json({
      answer: answer.output_text,
      tool_name: plan.tool_name,
      tool_arguments: plan.arguments || {},
      explore_next: exploreNext(domain),
      schema_cards: schemaCards(toolResult, domain)
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
