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

console.log('🚀 Portal starting...');
console.log('MCP_SERVER_URL:', MCP_SERVER_URL);
console.log('MODEL:', OPENAI_MODEL);

const client = new OpenAI({ apiKey: OPENAI_API_KEY });
const app = express();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

let mcpRequestId = 1;

function parseMcpResponseBody(rawText) {
  const trimmed = rawText.trim();

  if (!trimmed) {
    throw new Error('Empty MCP response body.');
  }

  if (trimmed.startsWith('{')) {
    return JSON.parse(trimmed);
  }

  const dataLines = trimmed
    .split('\n')
    .filter((line) => line.startsWith('data: '))
    .map((line) => line.slice(6).trim())
    .filter(Boolean);

  if (!dataLines.length) {
    throw new Error(`Unable to parse MCP response body: ${trimmed}`);
  }

  return JSON.parse(dataLines[dataLines.length - 1]);
}

async function callMcp(method, params = {}) {
  if (!MCP_SERVER_URL) {
    throw new Error('Missing MCP_SERVER_URL in server environment.');
  }

  const body = {
    jsonrpc: '2.0',
    id: mcpRequestId++,
    method,
    params
  };

  const response = await fetch(MCP_SERVER_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream'
    },
    body: JSON.stringify(body)
  });

  const rawText = await response.text();

  if (!response.ok) {
    throw new Error(`MCP ${method} failed with ${response.status}: ${rawText}`);
  }

  const parsed = parseMcpResponseBody(rawText);

  if (parsed.error) {
    throw new Error(
      `MCP ${method} error: ${parsed.error.message || JSON.stringify(parsed.error)}`
    );
  }

  return parsed.result;
}

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

  const fenced = text.match(/```json\s*([\s\S]*?)\s*```/i);
  if (fenced?.[1]) {
    const parsed = safeJsonParse(fenced[1]);
    if (parsed) return parsed;
  }

  const firstBrace = text.indexOf('{');
  const lastBrace = text.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    const slice = text.slice(firstBrace, lastBrace + 1);
    const parsed = safeJsonParse(slice);
    if (parsed) return parsed;
  }

  return null;
}

function simplifyToolListForPlanner(tools) {
  return tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    inputSchema: tool.inputSchema
  }));
}

// ✅ Cache MCP tools to avoid repeated tools/list calls and reduce 429s
let cachedTools = null;
let cachedToolsAt = 0;
const TOOLS_CACHE_MS = 5 * 60 * 1000; // 5 minutes

async function getCachedTools() {
  const now = Date.now();

  if (cachedTools && (now - cachedToolsAt) < TOOLS_CACHE_MS) {
    console.log('⚡ Using cached MCP tools');
    return cachedTools;
  }

  console.log('🔄 Fetching MCP tools from server...');
  const toolsListResult = await callMcp('tools/list', {});
  const tools = toolsListResult?.tools ?? [];

  if (!Array.isArray(tools) || tools.length === 0) {
    throw new Error('No tools returned from MCP server.');
  }

  cachedTools = tools;
  cachedToolsAt = now;
  return tools;
}

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'star-ai-intelligence-portal',
    has_openai_key: Boolean(OPENAI_API_KEY),
    has_mcp_server_url: Boolean(MCP_SERVER_URL),
    mcp_server_url: MCP_SERVER_URL || null,
    model: OPENAI_MODEL,
    tools_cached: Boolean(cachedTools),
    tools_cache_age_ms: cachedToolsAt ? Date.now() - cachedToolsAt : null
  });
});

app.post('/api/chat', async (req, res) => {
  try {
    if (!OPENAI_API_KEY) {
      return res.status(500).json({ error: 'Missing OPENAI_API_KEY in server environment.' });
    }

    if (!MCP_SERVER_URL) {
      return res.status(500).json({ error: 'Missing MCP_SERVER_URL in server environment.' });
    }

    const message = req.body?.message;

    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'A string "message" is required.' });
    }

    console.log('💬 User:', message);

    // 1) Use cached tools
    const tools = await getCachedTools();

    // 2) Ask OpenAI to choose the best MCP tool and arguments
    const plannerPrompt = `
You are choosing the best MCP tool for a STAR Automotive API Intelligence portal.

User request:
${message}

Available tools:
${JSON.stringify(simplifyToolListForPlanner(tools), null, 2)}

Return ONLY valid JSON with this exact shape:
{
  "tool_name": "string",
  "arguments": { }
}

Rules:
- Choose exactly one tool.
- Arguments must match the tool schema as closely as possible.
- Use empty object {} when no arguments are needed.
- Do not include markdown fences.
`;

    const plannerResponse = await client.responses.create({
      model: OPENAI_MODEL,
      input: plannerPrompt
    });

    const plannerText = plannerResponse.output_text || '';
    const plan = extractJsonObject(plannerText);

    if (!plan || typeof plan.tool_name !== 'string' || typeof plan.arguments !== 'object') {
      throw new Error(`Unable to parse tool selection JSON from model output: ${plannerText}`);
    }

    console.log('🧭 Selected MCP tool:', plan.tool_name, 'args:', plan.arguments);

    // 3) Call the chosen MCP tool directly
    const toolResult = await callMcp('tools/call', {
      name: plan.tool_name,
      arguments: plan.arguments || {}
    });

    // 4) Ask OpenAI to turn the raw MCP result into a polished answer
    const answerPrompt = `
You are a STAR Automotive API Intelligence assistant.

User request:
${message}

Tool used:
${plan.tool_name}

Tool arguments:
${JSON.stringify(plan.arguments || {}, null, 2)}

Raw MCP tool result:
${JSON.stringify(toolResult, null, 2)}

Write a clear, concise answer for the user.
If the tool result already fully answers the question, summarize it neatly.
`;

    const answerResponse = await client.responses.create({
      model: OPENAI_MODEL,
      input: answerPrompt
    });

    const answer = answerResponse.output_text || 'No response returned.';

    console.log('✅ Answer generated');

    return res.json({
      answer,
      tool_name: plan.tool_name,
      tool_arguments: plan.arguments || {},
      planner_response_id: plannerResponse.id || null,
      answer_response_id: answerResponse.id || null
    });
  } catch (e) {
    console.error('❌ AI request failed FULL ERROR:', e);

    return res.status(500).json({
      error: e?.error?.message || e?.message || 'AI request failed'
    });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🌐 STAR AI Portal running on 0.0.0.0:${PORT}`);
  console.log(`Health: http://0.0.0.0:${PORT}/health`);
});
