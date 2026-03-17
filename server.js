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

console.log('Portal MCP_SERVER_URL at startup:', MCP_SERVER_URL);
console.log('Portal model at startup:', OPENAI_MODEL);

const client = new OpenAI({ apiKey: OPENAI_API_KEY });
const app = express();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'star-ai-intelligence-portal',
    has_openai_key: Boolean(OPENAI_API_KEY),
    has_mcp_server_url: Boolean(MCP_SERVER_URL),
    mcp_server_url: MCP_SERVER_URL || null,
    model: OPENAI_MODEL
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

    console.log('Calling OpenAI with MCP server URL:', MCP_SERVER_URL);

    const response = await client.responses.create({
      model: OPENAI_MODEL,
      input: [
        {
          role: 'system',
          content: 'You are an assistant for STAR Automotive API Intelligence.'
        },
        {
          role: 'user',
          content: message
        }
      ],
      tools: [
        {
          type: 'mcp',
          server_label: 'star-mcp',
          server_description: 'STAR automotive metadata MCP server',
          server_url: MCP_SERVER_URL,
          require_approval: 'never'
        }
      ]
    });

    const answer =
      response.output_text ||
      'No response returned.';

    return res.json({
      answer,
      response_id: response.id || null
    });
  } catch (e) {
    console.error('AI request failed FULL ERROR:', e);

    return res.status(500).json({
      error: e?.error?.message || e?.message || 'AI request failed'
    });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`STAR AI Portal running on 0.0.0.0:${PORT}`);
  console.log(`Health: http://0.0.0.0:${PORT}/health`);
});
