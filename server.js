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

const client = new OpenAI({ apiKey: OPENAI_API_KEY });
const app = express();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

let mcpRequestId = 1;

/* ---------------- MCP HELPERS ---------------- */

function parseMcp(text) {
  if (text.startsWith('{')) return JSON.parse(text);

  const line = text.split('\n').find(l => l.startsWith('data:'));
  if (!line) throw new Error('Invalid MCP response');

  return JSON.parse(line.replace('data:', '').trim());
}

async function callMcp(method, params = {}) {
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
  return parsed.result;
}

async function callTool(name, args = {}) {
  return callMcp('tools/call', { name, arguments: args });
}

/* ---------------- INTENT DETECTION ---------------- */

function detectDomain(message) {
  const m = message.toLowerCase();

  if (m.includes('multi') || m.includes('inspection') || m.includes('mpi')) {
    return 'multi-point-inspection';
  }
  if (m.includes('appointment')) return 'appointment';

  return null;
}

function isOverview(message) {
  const m = message.toLowerCase();
  return m.includes('overview') || m.includes('what is') || m.includes('explain');
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
  const schemas = result?.schemas || result?.structuredContent?.schemas || [];

  return schemas.slice(0, 6).map(s => ({
    title: s.name,
    description: "View schema details",
    prompt: `Show schema ${s.name} in the ${domain} domain`
  }));
}

/* ---------------- ROUTES ---------------- */

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    model: OPENAI_MODEL,
    mcp: MCP_SERVER_URL
  });
});

app.post('/api/chat', async (req, res) => {
  try {
    const message = req.body.message;

    const domain = detectDomain(message);

    /* -------- OVERVIEW MODE (Claude-like) -------- */
    if (domain && isOverview(message)) {

      const overview = await callTool('getApiOverview', { domain_name: domain });
      const ops = await callTool('listOperations', { domain_name: domain });
      const schemas = await callTool('listSchemas', { domain_name: domain });

      const prompt = `
You are an expert automotive API architect.

Explain this domain clearly and richly.

User request:
${message}

Data:
Overview: ${JSON.stringify(overview)}
Operations: ${JSON.stringify(ops)}
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
- End with "Explore next" suggestions
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

    const tool = message.includes('schema')
      ? 'listSchemas'
      : 'listDomains';

    const toolResult = await callTool(tool, {});

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
      tool_arguments: {},
      explore_next: exploreNext(domain),
      schema_cards: schemaCards(toolResult, domain)
    });

  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => {
  console.log('🚀 Server running on port', PORT);
});
