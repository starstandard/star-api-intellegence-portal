
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
const MODEL = process.env.OPENAI_MODEL || "gpt-5";

const client = new OpenAI({ apiKey: OPENAI_API_KEY });
const app = express();

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.get("/health", (_, res) => {
  res.json({ status: "ok", service: "star-ai-intelligence-portal" });
});

app.post("/api/chat", async (req, res) => {
  try {
    const message = req.body.message;

    const response = await client.responses.create({
      model: MODEL,
      input: [
        { role: "system", content: "You are an assistant for STAR Automotive API Intelligence." },
        { role: "user", content: message }
      ],
      tools: [
        {
          type: "mcp",
          server_label: "star-mcp",
          server_url: MCP_SERVER_URL,
          require_approval: "never"
        }
      ]
    });

    res.json({ answer: response.output_text });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "AI request failed" });
  }
});

app.listen(PORT, () => {
  console.log("STAR AI Portal running on port", PORT);
});
