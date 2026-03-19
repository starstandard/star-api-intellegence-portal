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
  const trimmed = text.trim();

  if (!trimmed) {
    throw new Error('Empty MCP response');
  }

  if (trimmed.startsWith('{')) {
    return JSON.parse(trimmed);
  }

  const line = trimmed
    .split('\n')
    .find((l) => l.startsWith('data:'));

  if (!line) {
    throw new Error(`Invalid MCP response: ${trimmed}`);
  }

  return JSON.parse(line.replace('data:', '').trim());
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

/* ---------------- TOOL CALL CACHE ---------------- */

const toolCallCache = new Map();
const TOOL_CALL_CACHE_MS = 5 * 60 * 1000; // 5 minutes

function makeToolCacheKey(name, args = {}) {
  return `${name}:${JSON.stringify(args)}`;
}

async function callTool(name, args = {}) {
  return callMcp('tools/call', { name, arguments: args });
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

/* ---------------- INTENT DETECTION ---------------- */

function detectDomain(message) {
  const m = String(message || '').toLowerCase();

  if (m.includes('multi') || m.includes('inspection') || m.includes('mpi')) {
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
    m.includes('summary')
  );
}

/* ---------------- UI HELPERS ---------------- */

function exploreNext(domain) {
  if (!domain) return [];

  return [
    `List operations for the ${domain} domain`,
    `List schemas for the ${domain} domain`,
    `Show example endpoints for the ${domain} domain`
  ];
}

function schemaCards(result, domain) {
  const schemas =
    result?.schemas ||
    result?.structuredContent?.schemas ||
    [];

  return schemas.slice(0, 6).map((s) => ({
    title: s.name,
    description: 'View schema details',
    prompt: `Show schema ${s.name} in the ${domain} domain`
  }));
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

    /* -------- OVERVIEW MODE (Claude-like) -------- */
    if (domain && isOverview(message)) {
      // Lighter version: only 2 core MCP calls instead of 3
      // You can re-add listOperations later if rate limits settle down.
      const overview = await callToolCached('getApiOverview', { domain_name: domain });
      const schemas = await callToolCached('listSchemas', { domain_name: domain });

      const prompt = `
You are an expert automotive API architect.

Explain this domain clearly and richly.

User request:
${message}

Data:
Overview: ${JSON.stringify(overview)}
Schemas: ${JSON.stringify(schemas)}

Structure your answer:

1. What it is
2. Core concepts
3. Main resources
4. Workflow
5. Key schemas
6. Example endpoints

IMPORTANT:
- Do NOT ask follow-up questions
- Do NOT say "Would you like me to..."
- End with actionable next steps
- Be concise but useful
`;

      const response = await client.responses.create({
        model: OPENAI_MODEL,
        input: prompt
      });

      return res.json({
        answer: response.output_text,
        tool_name: 'overview_pipeline',
        tool_arguments: { domain },
        explore_next: exploreNext(domain),
        schema_cards: schemaCards(schemas, domain)
      });
    }

    /* -------- DEFAULT TOOL FLOW -------- */

    const tool = message.toLowerCase().includes('schema')
      ? 'listSchemas'
      : 'listDomains';

    const toolArgs =
      tool === 'listSchemas' && domain
        ? { domain_name: domain }
        : {};

    const toolResult = await callToolCached(tool, toolArgs);

    const answer = await client.responses.create({
      model: OPENAI_MODEL,
      input: `
User: ${message}
Data: ${JSON.stringify(toolResult)}

Explain clearly and concisely.
Do NOT ask "would you like".
Give actionable info.
`
    });

    return res.json({
      answer: answer.output_text,
      tool_name: tool,
      tool_arguments: toolArgs,
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
