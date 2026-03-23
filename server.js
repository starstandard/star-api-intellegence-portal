// STAR V4 SERVER (Capability Navigator Enabled)
import 'dotenv/config';
import express from 'express';
import OpenAI from 'openai';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MCP_SERVER_URL = process.env.MCP_SERVER_URL;
const MODEL = process.env.OPENAI_MODEL || 'gpt-5';

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

/* ---------- MCP ---------- */
let id = 1;

async function callMcp(method, params={}) {
  const res = await fetch(MCP_SERVER_URL,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({jsonrpc:'2.0', id:id++, method, params})
  });
  const json = await res.json();
  if(json.error) throw new Error(JSON.stringify(json.error));
  return json.result;
}

async function tool(name,args={}) {
  return callMcp('tools/call',{name, arguments:args});
}

/* ---------- HELPERS ---------- */

function detectDomain(m){
  m=m.toLowerCase();
  if(m.includes('inspection')||m.includes('mpi')) return 'multi-point-inspection';
  if(m.includes('appointment')) return 'appointment';
  return null;
}

function wantsCapabilities(m){
  m=m.toLowerCase();
  return m.includes('capabilities')||m.includes('service advisor');
}

/* ---------- CAPABILITY MODE ---------- */

async function buildCapabilities(domain){
  const ops = await tool('listOperations',{domain_name:domain});

  const response = await client.responses.create({
    model: MODEL,
    input: `Group these operations into business capabilities for a dealership service advisor. Return JSON:
{ "answer": "...", "capability_cards":[{ "title":"","description":"","identifier":"","prompt":"" }] }

DATA:
${JSON.stringify(ops)}`
  });

  return JSON.parse(response.output_text);
}

/* ---------- ROUTE ---------- */

app.post('/api/chat', async (req,res)=>{
  try{
    const message = req.body.message;
    const domain = detectDomain(message);

    if(domain && wantsCapabilities(message)){
      const nav = await buildCapabilities(domain);
      return res.json({
        answer: nav.answer,
        capability_cards: nav.capability_cards
      });
    }

    const result = await tool('listDomains',{});

    const answer = await client.responses.create({
      model: MODEL,
      input: `User:${message}\nData:${JSON.stringify(result)}`
    });

    res.json({
      answer: answer.output_text
    });

  }catch(e){
    res.status(500).json({error:e.message});
  }
});

app.listen(PORT,()=>console.log('V4 running',PORT));
