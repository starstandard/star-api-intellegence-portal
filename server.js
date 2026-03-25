import 'dotenv/config';
import express from 'express';
import OpenAI from 'openai';
import path from 'node:path';
import fs from 'node:fs';
import yaml from 'js-yaml';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const MCP_SERVER_URL = process.env.MCP_SERVER_URL;
const MODEL = process.env.OPENAI_MODEL || 'gpt-5';
const OPENAPI_SPEC_PATH = process.env.OPENAPI_SPEC_PATH || '';

if (!OPENAI_API_KEY) console.error('Missing OPENAI_API_KEY');
if (!MCP_SERVER_URL) console.error('Missing MCP_SERVER_URL');

const client = new OpenAI({ apiKey: OPENAI_API_KEY });
const app = express();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

/* --------------------------------------------------
 * TIMEOUTS
 * -------------------------------------------------- */

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

/* --------------------------------------------------
 * MCP
 * -------------------------------------------------- */

let rpcId = 1;

function parseMcpResponse(text, contentType = '') {
  const trimmed = String(text || '').trim();
  if (!trimmed) throw new Error('Empty MCP response');

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

function isRateLimitErrorMessage(msg = '') {
  const m = String(msg).toLowerCase();
  return (
    m.includes('too many requests') ||
    m.includes('rate limit') ||
    m.includes('rate-limited') ||
    m.includes('being rate limited') ||
    m.includes('"code":429') ||
    m.includes(' 429') ||
    m.includes('429')
  );
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
        Accept: 'application/json, text/event-stream'
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: rpcId++,
        method,
        params
      })
    }),
    20000,
    `MCP ${method} request`
  );

  const contentType = response.headers.get('content-type') || '';
  const text = await withTimeout(response.text(), 10000, `MCP ${method} body read`);

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

/* --------------------------------------------------
 * CACHE
 * -------------------------------------------------- */

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

/* --------------------------------------------------
 * NORMALIZATION / INTENT HELPERS
 * -------------------------------------------------- */

const DOMAIN_HINTS = {
  appointment: {
    resourceAliases: ['/appointments', '/appointment'],
    keywords: ['appointment', 'requested service', 'time slot', 'dealer', 'service advisor'],
    schemaKeywords: ['appointment', 'requestedservice', 'timeslot', 'vehicle', 'party', 'dealer', 'advisor']
  },
  'multi-point-inspection': {
    resourceAliases: [
      '/inspections',
      '/inspection',
      '/multipoint-inspections',
      '/multi-point-inspections',
      '/multi_point_inspections'
    ],
    keywords: ['inspection', 'finding', 'recommendation', 'approval', 'mpi', 'media'],
    schemaKeywords: [
      'inspection',
      'finding',
      'recommendation',
      'approval',
      'media',
      'condition',
      'manifest',
      'line',
      'status'
    ]
  }
};

function normalizeLooseText(message = '') {
  return String(message)
    .toLowerCase()
    .replace(/inspectios/g, 'inspections')
    .replace(/opertion/g, 'operation')
    .replace(/operationonly/g, 'operation only')
    .replace(/subapi/g, 'sub api')
    .replace(/multipoint-inspection/g, 'multi-point-inspection')
    .replace(/multipoint inspections/g, 'multi-point-inspections')
    .replace(/\s+/g, ' ')
    .trim();
}

function canonicalizeEndpointAlias(endpoint = '') {
  return String(endpoint)
    .toUpperCase()
    .replace('/MULTIPOINT-INSPECTIONS', '/INSPECTIONS')
    .replace('/MULTI-POINT-INSPECTIONS', '/INSPECTIONS')
    .replace('/MULTI_POINT_INSPECTIONS', '/INSPECTIONS')
    .replace('/INSPECTIOS', '/INSPECTIONS')
    .trim();
}

function normalizeEndpointText(endpoint = '') {
  return String(endpoint)
    .toUpperCase()
    .replace('/INSPECTIOS', '/INSPECTIONS')
    .replace('/INSPECTIO', '/INSPECTION')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeEndpointTitle(endpointTitle = '') {
  return canonicalizeEndpointAlias(normalizeEndpointText(endpointTitle));
}

function normalizePathForMatch(routePath = '') {
  return canonicalizeEndpointAlias(String(routePath).toUpperCase())
    .replace(/\{[^}]+\}/g, '{PARAM}')
    .replace(/\/+$/, '');
}

function detectDomainFromEndpoint(message = '') {
  const m = normalizeLooseText(message);

  if (DOMAIN_HINTS['multi-point-inspection'].resourceAliases.some((alias) => m.includes(alias))) {
    return 'multi-point-inspection';
  }

  if (DOMAIN_HINTS.appointment.resourceAliases.some((alias) => m.includes(alias))) {
    return 'appointment';
  }

  return null;
}

function detectDomain(message = '') {
  const m = normalizeLooseText(message);

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

  const endpointDomain = detectDomainFromEndpoint(message);
  if (endpointDomain) return endpointDomain;

  return null;
}

function normalizeAudience(audience = '') {
  const a = String(audience).toLowerCase();
  if (a === 'technical') return 'technical';
  if (a === 'architecture') return 'architecture';
  return 'business';
}

function wantsCapabilities(message = '') {
  const m = normalizeLooseText(message);
  return (
    m.includes('capabilities') ||
    m.includes('business capabilities') ||
    m.includes('service advisor') ||
    m.includes('service lane') ||
    m.includes('advisor-facing') ||
    m.includes('daily workflow')
  );
}

function wantsOverview(message = '') {
  const m = normalizeLooseText(message);
  return (
    m.includes('overview') ||
    m.includes('what is') ||
    m.includes('summarize') ||
    m.includes('summary') ||
    m.includes('tell me about') ||
    m.includes('executive overview')
  );
}

function wantsRichApiDescription(message = '') {
  const m = normalizeLooseText(message);
  return (
    m.includes('rich description') ||
    m.includes('detailed description') ||
    m.includes('describe the api') ||
    m.includes('describe this api') ||
    m.includes('full description') ||
    m.includes('deep description') ||
    m.includes('rich api description')
  );
}

function wantsSubApiDesign(message = '') {
  const m = normalizeLooseText(message);
  return (
    m.includes('sub-api') ||
    m.includes('sub api') ||
    m.includes('get operations only') ||
    m.includes('operation only') ||
    m.includes('only for the sub-api') ||
    m.includes('only for the sub api') ||
    m.includes('read-only api') ||
    m.includes('read only api') ||
    m.includes('derive an api') ||
    m.includes('build an api from') ||
    m.includes('build a sub-api') ||
    m.includes('propose a sub-api') ||
    (m.includes('get /') && m.includes('sub api'))
  );
}

function wantsSchemas(message = '') {
  return normalizeLooseText(message).includes('schema');
}

function wantsEndpoints(message = '') {
  const m = normalizeLooseText(message);
  return (
    m.includes('endpoint') ||
    m.includes('endpoints') ||
    m.includes('operation') ||
    m.includes('operations') ||
    m.includes('url') ||
    m.includes('path') ||
    /\b(get|post|put|patch|delete)\s+\//i.test(message)
  );
}

function extractRequestedSchemaName(message = '') {
  const text = String(message || '');
  const patterns = [
    /show schema\s+(.+?)(?:\s+in\s+the\s+.+?api)?$/i,
    /explain schema\s+(.+?)(?:\s+in\s+the\s+.+?api)?$/i,
    /schema\s+(.+?)(?:\s+details)?(?:\s+in\s+the\s+.+?api)?$/i
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match && match[1]) return match[1].trim();
  }

  return null;
}

function extractRequestedEndpoint(message = '') {
  const text = String(message || '').trim();
  const match = text.match(/(GET|POST|PUT|PATCH|DELETE)\s+\/[A-Za-z0-9/_{}-]+/i);
  if (!match) return null;
  return canonicalizeEndpointAlias(normalizeEndpointText(match[0]));
}

function extractOperations(result) {
  return result?.operations || result?.structuredContent?.operations || [];
}

function extractSchemas(result) {
  return result?.schemas || result?.structuredContent?.schemas || [];
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
    .map((s) => ({ title: s.title, body: s.body.join('\n').trim() }))
    .filter((s) => s.title || s.body);
}

/* --------------------------------------------------
 * OPENAPI RAG LAYER
 * -------------------------------------------------- */

function resolveCandidateSpecPaths() {
  const candidates = [
    OPENAPI_SPEC_PATH,
    path.join(__dirname, 'openapi_monolith.yaml'),
    path.join(__dirname, 'openapi_enriched.yaml'),
    path.join(__dirname, 'openapi.yaml'),
    path.join(__dirname, 'openapi.yml'),
    path.join(__dirname, 'schemas', 'openapi_monolith.yaml'),
    path.join(__dirname, 'schemas', 'openapi_enriched.yaml')
  ].filter(Boolean);

  return [...new Set(candidates)];
}

function loadOpenApiSpec() {
  for (const candidate of resolveCandidateSpecPaths()) {
    try {
      if (!fs.existsSync(candidate)) continue;
      const raw = fs.readFileSync(candidate, 'utf8');
      const parsed = yaml.load(raw);
      if (parsed && typeof parsed === 'object') {
        console.log(`✅ Loaded OpenAPI spec from ${candidate}`);
        return { spec: parsed, sourcePath: candidate };
      }
    } catch (err) {
      console.warn(`⚠️ Failed to load OpenAPI spec from ${candidate}: ${err.message}`);
    }
  }

  console.warn('⚠️ No OpenAPI spec loaded. Falling back to MCP + built-in content.');
  return { spec: null, sourcePath: null };
}

const OPENAPI_STATE = loadOpenApiSpec();

function getOpenApiSpec() {
  return OPENAPI_STATE.spec;
}

function operationMatchesDomain(domain, routePath, op) {
  const hints = DOMAIN_HINTS[domain];
  if (!hints) return false;

  const haystack = [
    routePath,
    ...(Array.isArray(op?.tags) ? op.tags : []),
    op?.summary || '',
    op?.description || '',
    op?.operationId || ''
  ]
    .join(' ')
    .toLowerCase();

  return hints.keywords.some((kw) => haystack.includes(kw));
}

function getOpenApiOperationsForDomain(domain) {
  const spec = getOpenApiSpec();
  if (!spec?.paths) return [];

  const ops = [];
  for (const [routePath, pathItem] of Object.entries(spec.paths)) {
    for (const method of ['get', 'post', 'put', 'patch', 'delete']) {
      const op = pathItem?.[method];
      if (!op) continue;
      if (!operationMatchesDomain(domain, routePath, op)) continue;

      ops.push({
        method: method.toUpperCase(),
        path: routePath,
        summary: op.summary || '',
        description: op.description || '',
        operationId: op.operationId || '',
        tags: Array.isArray(op.tags) ? op.tags : [],
        raw: op
      });
    }
  }

  return ops;
}

function getOpenApiSchemaMap() {
  const spec = getOpenApiSpec();
  return spec?.components?.schemas || {};
}

function collectSchemaNamesFromRefs(value, acc = new Set()) {
  if (!value || typeof value !== 'object') return acc;

  if (typeof value.$ref === 'string') {
    const match = value.$ref.match(/#\/components\/schemas\/(.+)$/);
    if (match?.[1]) acc.add(match[1]);
  }

  for (const child of Object.values(value)) {
    if (child && typeof child === 'object') collectSchemaNamesFromRefs(child, acc);
  }

  return acc;
}

function getOpenApiSchemasForDomain(domain) {
  const schemaMap = getOpenApiSchemaMap();
  const schemaEntries = Object.entries(schemaMap);
  if (!schemaEntries.length) return [];

  const matched = new Set();
  const hints = DOMAIN_HINTS[domain];

  for (const [name, schema] of schemaEntries) {
    const haystack = `${name} ${schema?.title || ''} ${schema?.description || ''}`.toLowerCase();
    if (hints?.schemaKeywords?.some((kw) => haystack.includes(kw))) matched.add(name);
  }

  for (const op of getOpenApiOperationsForDomain(domain)) {
    collectSchemaNamesFromRefs(op.raw, matched);
  }

  return Array.from(matched).map((name) => ({
    name,
    description: schemaMap[name]?.description || '',
    raw: schemaMap[name]
  }));
}

function findOpenApiSchemaByName(schemaName) {
  const schemaMap = getOpenApiSchemaMap();
  const target = String(schemaName || '').trim().toLowerCase();
  if (!target) return null;

  for (const [name, schema] of Object.entries(schemaMap)) {
    if (name.toLowerCase() === target) return { name, raw: schema };
  }

  for (const [name, schema] of Object.entries(schemaMap)) {
    if (name.toLowerCase().includes(target) || target.includes(name.toLowerCase())) {
      return { name, raw: schema };
    }
  }

  return null;
}

function findOpenApiEndpoint(endpointTitle) {
  const spec = getOpenApiSpec();
  if (!spec?.paths) return null;

  const normalizedTarget = normalizeEndpointTitle(endpointTitle);
  const spaceIndex = normalizedTarget.indexOf(' ');
  if (spaceIndex === -1) return null;

  const method = normalizedTarget.slice(0, spaceIndex);
  const routePath = normalizedTarget.slice(spaceIndex + 1);
  const targetNormalized = normalizePathForMatch(routePath);

  for (const [p, pathItem] of Object.entries(spec.paths)) {
    const candidateNormalized = normalizePathForMatch(p);
    if (candidateNormalized !== targetNormalized) continue;

    const op = pathItem?.[method.toLowerCase()];
    if (!op) continue;

    return {
      method,
      path: p,
      details: op
    };
  }

  return null;
}

function dereferenceSchemaRef(ref) {
  if (typeof ref !== 'string') return null;
  const match = ref.match(/#\/components\/schemas\/(.+)$/);
  if (!match?.[1]) return null;
  const name = match[1];
  const schema = getOpenApiSchemaMap()[name];
  if (!schema) return null;
  return { name, raw: schema };
}

function extractRelatedSchemasFromOperation(op) {
  const found = collectSchemaNamesFromRefs(op, new Set());
  return Array.from(found).map((name) => {
    const schema = getOpenApiSchemaMap()[name];
    return {
      title: name,
      description: schema?.description || 'Related OpenAPI schema',
      prompt: `Show schema ${name}`
    };
  });
}

function extractParametersFromOperation(op) {
  const parameters = Array.isArray(op?.parameters) ? op.parameters : [];
  return parameters.map((p) => ({
    name: p.name || '',
    in: p.in || '',
    required: Boolean(p.required),
    description: p.description || '',
    schema: p.schema || null,
    example: p.example ?? p.schema?.example ?? null
  }));
}

function findPrimaryJsonContent(contentObj = {}) {
  if (!contentObj || typeof contentObj !== 'object') return null;
  return (
    contentObj['application/json'] ||
    contentObj['application/*+json'] ||
    Object.values(contentObj)[0] ||
    null
  );
}

function buildSampleFromSchema(schema, depth = 0) {
  if (!schema || depth > 3) return null;

  if (schema.example !== undefined) return schema.example;
  if (Array.isArray(schema.examples) && schema.examples.length) return schema.examples[0];

  if (schema.$ref) {
    const deref = dereferenceSchemaRef(schema.$ref);
    return deref ? buildSampleFromSchema(deref.raw, depth + 1) : null;
  }

  if (schema.type === 'object' || schema.properties) {
    const out = {};
    for (const [key, prop] of Object.entries(schema.properties || {})) {
      const sample = buildSampleFromSchema(prop, depth + 1);
      out[key] = sample !== null ? sample : '<value>';
    }
    return out;
  }

  if (schema.type === 'array') {
    const item = buildSampleFromSchema(schema.items, depth + 1);
    return [item !== null ? item : '<item>'];
  }

  if (schema.enum?.length) return schema.enum[0];
  if (schema.format === 'date-time') return '2026-03-25T12:00:00Z';
  if (schema.format === 'date') return '2026-03-25';
  if (schema.format === 'uri') return 'https://example.com/resource';

  switch (schema.type) {
    case 'string':
      return schema.title ? `<${schema.title}>` : '<string>';
    case 'integer':
    case 'number':
      return 0;
    case 'boolean':
      return true;
    default:
      return null;
  }
}

function extractRequestExampleFromOperation(op) {
  const content = findPrimaryJsonContent(op?.requestBody?.content);
  if (!content) return null;

  if (content.example !== undefined) return content.example;

  if (content.examples && typeof content.examples === 'object') {
    const first = Object.values(content.examples)[0];
    if (first?.value !== undefined) return first.value;
  }

  return buildSampleFromSchema(content.schema);
}

function extractResponseExampleFromOperation(op) {
  const responses = op?.responses || {};
  const preferred = responses['200'] || responses['201'] || responses['202'] || Object.values(responses)[0];
  if (!preferred) return null;

  const content = findPrimaryJsonContent(preferred.content);
  if (!content) return null;

  if (content.example !== undefined) return content.example;

  if (content.examples && typeof content.examples === 'object') {
    const first = Object.values(content.examples)[0];
    if (first?.value !== undefined) return first.value;
  }

  return buildSampleFromSchema(content.schema);
}

function extractResponseSchemaCardFromOperation(op) {
  const responses = op?.responses || {};
  const preferred = responses['200'] || responses['201'] || responses['202'] || Object.values(responses)[0];
  if (!preferred) return [];

  const content = findPrimaryJsonContent(preferred.content);
  if (!content?.schema) return [];

  const found = collectSchemaNamesFromRefs(content.schema, new Set());
  return Array.from(found).map((name) => {
    const schema = getOpenApiSchemaMap()[name];
    return {
      title: name,
      description: schema?.description || 'Response schema',
      prompt: `Show schema ${name}`
    };
  });
}

/* --------------------------------------------------
 * SAFE MCP + OPENAPI ENRICHMENT
 * -------------------------------------------------- */

async function safeListSchemas(domain, limit = 6) {
  try {
    const openApiSchemas = getOpenApiSchemasForDomain(domain);
    if (openApiSchemas.length) {
      return { schemas: openApiSchemas.slice(0, limit) };
    }
  } catch {}

  try {
    return await toolCached('listSchemas', { domain_name: domain, limit });
  } catch {
    return { schemas: [] };
  }
}

async function safeListOperations(domain, limit = 6) {
  try {
    const openApiOps = getOpenApiOperationsForDomain(domain);
    if (openApiOps.length) {
      return { operations: openApiOps.slice(0, limit) };
    }
  } catch {}

  try {
    return await toolCached('listOperations', { domain_name: domain, limit });
  } catch {
    return { operations: [] };
  }
}

async function safeGetOverview(domain) {
  try {
    return await toolCached('getApiOverview', { domain_name: domain });
  } catch {
    return null;
  }
}

function deriveEndpointCards(result, domain) {
  return extractOperations(result).slice(0, 8).map((op) => ({
    title: `${op.method || 'GET'} ${op.path || ''}`.trim(),
    description: op.summary || op.description || 'API endpoint',
    prompt: `Explain ${(op.method || 'GET')} ${op.path || ''} for the ${domain || 'current'} API`
  }));
}

function deriveSchemaCards(result, domain) {
  return extractSchemas(result).slice(0, 16).map((s) => ({
    title: s.name,
    description: s.description || 'View schema details',
    prompt: `Show schema ${s.name} in the ${domain || 'current'} API`
  }));
}

/* --------------------------------------------------
 * RESPONSE BUILDERS
 * -------------------------------------------------- */

async function buildCapabilityResponse(domain, audience) {
  const builtInCards = getBuiltInCapabilityCards(domain);
  const workflowMap = getBuiltInWorkflowMap(domain);

  const [ops, schemas] = await Promise.all([
    safeListOperations(domain, 6),
    safeListSchemas(domain, 12)
  ]);

  const liveEndpointCards = deriveEndpointCards(ops, domain);
  const liveSchemaCards = deriveSchemaCards(schemas, domain);

  const endpointCards = liveEndpointCards.length ? liveEndpointCards : getBuiltInEndpointCards(domain);
  const schemaCards = liveSchemaCards.length ? liveSchemaCards : getBuiltInSchemaCards(domain);

  const intro = audience === 'technical'
    ? `1. Direct answer
Technical capability views for the ${domain} API are shown below.

2. Key details
Each capability includes technical notes, data model notes, and integration notes in addition to business meaning.

3. Useful next steps
Select a capability card to inspect technical details and implementation context.`
    : audience === 'architecture'
      ? `1. Direct answer
Architecture-oriented capability views for the ${domain} API are shown below.

2. Key details
Each capability includes context boundaries, data-model implications, and integration implications relevant to architecture work.

3. Useful next steps
Select a capability card to inspect architecture-oriented details and dependencies.`
      : `1. Direct answer
Business capabilities for the ${domain} API are shown below.

2. Key details
These capabilities are organized around the dealership workflow and include advisor-facing meaning, workflow stage, and customer impact.

3. Useful next steps
Select a capability card to go deeper into business guidance and next actions.`;

  return {
    answer: intro,
    sections: extractSections(intro),
    capability_cards: builtInCards,
    endpoint_cards: endpointCards,
    schema_cards: schemaCards,
    workflow_map: workflowMap,
    progressive: liveEndpointCards.length === 0 || liveSchemaCards.length === 0
  };
}

async function buildOverviewResponse(domain, message, audience) {
  const [overview, ops, schemas] = await Promise.all([
    safeGetOverview(domain),
    safeListOperations(domain, 8),
    safeListSchemas(domain, 16)
  ]);

  const builtInOverview = getAudienceOverview(domain, audience);

  let answer;
  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `You are a STAR automotive retail API architect.
Audience: ${audience}
The appointment API means dealer vehicle service appointments, not medical scheduling.
The multi-point-inspection API means vehicle inspection workflow.

User request:
${message}

Built-in domain overview:
${builtInOverview}

MCP or OpenAPI overview:
${JSON.stringify(overview)}

Operations:
${JSON.stringify(ops)}

Schemas:
${JSON.stringify(schemas)}

Write a rich structured answer that preserves the depth of the built-in domain overview while using retrieved data to enrich it where possible.`
      }),
      10000,
      'OpenAI overview generation'
    );

    answer = response.output_text;
  } catch {
    answer = builtInOverview;
  }

  const liveEndpointCards = deriveEndpointCards(ops, domain);
  const liveSchemaCards = deriveSchemaCards(schemas, domain);

  return {
    answer,
    sections: extractSections(answer),
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: liveEndpointCards.length ? liveEndpointCards : getBuiltInEndpointCards(domain),
    schema_cards: liveSchemaCards.length ? liveSchemaCards : getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: liveEndpointCards.length === 0 || liveSchemaCards.length === 0
  };
}

async function buildRichApiDescriptionResponse(domain, message, audience) {
  const [overview, ops, schemas] = await Promise.all([
    safeGetOverview(domain),
    safeListOperations(domain, 12),
    safeListSchemas(domain, 16)
  ]);

  const builtInOverview = getAudienceOverview(domain, audience);

  let answer;
  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `You are a STAR automotive retail API architect.
Audience: ${audience}

User request:
${message}

Domain:
${domain}

Built-in audience-aware description:
${builtInOverview}

Retrieved overview:
${JSON.stringify(overview)}

Operations:
${JSON.stringify(ops)}

Schemas:
${JSON.stringify(schemas)}

Produce a rich API description tailored to the audience.

Return:
1. Direct answer
2. Core responsibilities
3. Main resources and schemas
4. Workflow or lifecycle
5. Integration or architecture notes
6. Useful next steps`
      }),
      16000,
      'OpenAI rich API description generation'
    );

    answer = response.output_text;
  } catch {
    answer = builtInOverview;
  }

  const liveEndpointCards = deriveEndpointCards(ops, domain);
  const liveSchemaCards = deriveSchemaCards(schemas, domain);

  return {
    answer,
    sections: extractSections(answer),
    audience,
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: liveEndpointCards.length ? liveEndpointCards : getBuiltInEndpointCards(domain),
    schema_cards: liveSchemaCards.length ? liveSchemaCards : getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: liveEndpointCards.length === 0 || liveSchemaCards.length === 0
  };
}

async function buildSubApiResponse(domain, message, audience) {
  const ops = await safeListOperations(domain, 20);
  const schemas = await safeListSchemas(domain, 16);

  let answer;
  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `You are a STAR automotive retail API architect.
Audience: ${audience}

User request:
${message}

Domain:
${domain}

Operations:
${JSON.stringify(ops)}

Schemas:
${JSON.stringify(schemas)}

Design a focused sub-API based on the user's request.

Return:
1. Direct answer
2. Proposed resources or endpoints
3. Proposed schemas
4. Design notes
5. Suggested next steps`
      }),
      18000,
      'OpenAI sub-api generation'
    );

    answer = response.output_text;
  } catch {
    answer = `1. Direct answer
A focused sub-API can be derived for the ${domain} domain.

2. Proposed resources or endpoints
Start with a reduced set of read-oriented GET operations that expose the main collection and entity retrieval flows.

3. Proposed schemas
Reuse the core read models and supporting reference schemas needed for consumers.

4. Design notes
Prefer a narrow read-only surface, stable identifiers, and minimal cross-domain coupling.

5. Suggested next steps
List the GET operations for the domain and identify which schemas should be exposed in the reduced API.`;
  }

  const endpointCards = deriveEndpointCards(ops, domain);
  const schemaCards = deriveSchemaCards(schemas, domain);

  return {
    answer,
    sections: extractSections(answer),
    audience,
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: endpointCards.length ? endpointCards : getBuiltInEndpointCards(domain),
    schema_cards: schemaCards.length ? schemaCards : getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: false
  };
}

async function buildSchemaResponse(domain, message, audience) {
  const schemas = await safeListSchemas(domain, 16);
  const liveSchemaCards = deriveSchemaCards(schemas, domain);
  const schemaCards = liveSchemaCards.length ? liveSchemaCards : getBuiltInSchemaCards(domain);

  let answer;
  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `You are a STAR automotive retail API assistant.
Audience: ${audience}

User request:
${message}

Schemas:
${JSON.stringify(schemas)}

Write a concise answer with:
1. Direct answer
2. Key schema details
3. Useful next steps`
      }),
      9000,
      'OpenAI schema answer generation'
    );

    answer = response.output_text;
  } catch {
    answer = `1. Direct answer
Representative schemas for the ${domain} API are shown in the inspector.

2. Key schema details
These schema cards represent the primary object families and data structures that technical and business users typically need first.

3. Useful next steps
Open a schema card, switch to Technical mode for schema-oriented reading, or ask for a specific schema by name.`;
  }

  return {
    answer,
    sections: extractSections(answer),
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: [],
    schema_cards: schemaCards,
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: liveSchemaCards.length === 0
  };
}

async function buildSchemaDetailResponse(domain, schemaName, audience) {
  let liveSchema = null;

  const openApiSchema = findOpenApiSchemaByName(schemaName);
  if (openApiSchema) {
    liveSchema = openApiSchema.raw;
    schemaName = openApiSchema.name;
  } else {
    try {
      liveSchema = await toolCached('getSchema', {
        domain_name: domain,
        schema_name: schemaName
      });
    } catch {
      liveSchema = null;
    }
  }

  const builtIn = getBuiltInSchemaDetails(domain, schemaName);
  const rawSchema = liveSchema || getBuiltInRawSchema(domain, schemaName);
  const relatedSchemas = rawSchema
    ? Array.from(collectSchemaNamesFromRefs(rawSchema, new Set())).map((name) => ({
        title: name,
        description: getOpenApiSchemaMap()[name]?.description || 'Related schema',
        prompt: `Show schema ${name}`
      }))
    : [];

  let answer;
  if (rawSchema) {
    try {
      const response = await withTimeout(
        client.responses.create({
          model: MODEL,
          input: `You are a STAR automotive retail API assistant.
Audience: ${audience}

Schema name:
${schemaName}

Schema data:
${JSON.stringify(rawSchema)}

Write a structured explanation with:
1. Direct answer
2. Key schema details
3. Why it matters
4. Useful next steps`
        }),
        10000,
        'OpenAI schema detail generation'
      );

      answer = response.output_text;
    } catch {
      answer = builtIn?.body || `1. Direct answer
${schemaName} is a schema in the ${domain} API.

2. Key schema details
Detailed schema content could not be retrieved right now.

3. Useful next steps
Inspect related schema cards or ask for endpoints tied to this schema.`;
    }
  } else {
    answer = builtIn?.body || `1. Direct answer
${schemaName} is a schema in the ${domain} API.

2. Key schema details
Detailed schema content could not be retrieved right now.

3. Useful next steps
Inspect related schema cards or ask for endpoints tied to this schema.`;
  }

  return {
    answer,
    sections: extractSections(answer),
    audience,
    raw_schema: rawSchema,
    raw_schema_format: 'json',
    related_schema_cards: relatedSchemas,
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: getBuiltInEndpointCards(domain),
    schema_cards: getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: !liveSchema
  };
}

async function buildEndpointResponse(domain, message, audience) {
  const ops = await safeListOperations(domain, 8);
  const liveEndpointCards = deriveEndpointCards(ops, domain);
  const endpointCards = liveEndpointCards.length ? liveEndpointCards : getBuiltInEndpointCards(domain);

  let answer;
  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `You are a STAR automotive retail API assistant.
Audience: ${audience}

User request:
${message}

Operations:
${JSON.stringify(ops)}

Write a concise answer with:
1. Direct answer
2. Key endpoint details
3. Useful next steps`
      }),
      15000,
      'OpenAI endpoint answer generation'
    );

    answer = response.output_text;
  } catch {
    answer = `1. Direct answer
Representative endpoints for the ${domain} API are shown in the inspector.

2. Key endpoint details
These endpoints represent the main operation families for the domain and can be used to understand how clients interact with the API.

3. Useful next steps
Inspect the endpoint cards, switch to Technical mode for implementation-oriented reading, or ask for a specific operation, method, or path.`;
  }

  return {
    answer,
    sections: extractSections(answer),
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: endpointCards,
    schema_cards: getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: liveEndpointCards.length === 0
  };
}

async function buildEndpointDetailResponse(domain, endpointTitle, audience) {
  const openApiEndpoint = findOpenApiEndpoint(endpointTitle);
  const builtIn = getBuiltInEndpointDetail(domain, endpointTitle);

  let answer;
  let parameters = [];
  let relatedSchemaCards = [];
  let requestExample = null;
  let responseExample = null;

  if (openApiEndpoint) {
    const op = openApiEndpoint.details;
    parameters = extractParametersFromOperation(op);
    relatedSchemaCards = [
      ...extractRelatedSchemasFromOperation(op),
      ...extractResponseSchemaCardFromOperation(op)
    ].filter((item, idx, arr) => arr.findIndex((x) => x.title === item.title) === idx);
    requestExample = extractRequestExampleFromOperation(op);
    responseExample = extractResponseExampleFromOperation(op);

    answer = `1. Direct answer
${openApiEndpoint.method} ${openApiEndpoint.path} is a defined endpoint in the ${domain} API.

2. Key endpoint details
${op.summary || op.description || 'This endpoint is part of the domain API surface.'}

3. Why it matters
This endpoint is relevant to consumers who need direct access to this resource or workflow step.`;
  } else {
    answer = builtIn || `1. Direct answer
${endpointTitle} is an endpoint in the ${domain} API.

2. Key endpoint details
Detailed live endpoint content is not available right now.

3. Useful next steps
Inspect related schemas or try again later.`;
  }

  return {
    answer,
    sections: extractSections(answer),
    audience,
    endpoint_parameters: parameters,
    related_schema_cards: relatedSchemaCards,
    request_example: requestExample,
    response_example: responseExample,
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: getBuiltInEndpointCards(domain),
    schema_cards: getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: !openApiEndpoint
  };
}

/* --------------------------------------------------
 * ROUTES
 * -------------------------------------------------- */

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'star-ai-intelligence-portal',
    model: MODEL,
    has_openai_key: Boolean(OPENAI_API_KEY),
    has_mcp_server_url: Boolean(MCP_SERVER_URL),
    has_openapi_spec: Boolean(getOpenApiSpec()),
    openapi_source: OPENAPI_STATE.sourcePath,
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

    const message = req.body?.message;
    const audience = normalizeAudience(req.body?.audience || 'business');

    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'A string "message" is required.' });
    }

    const domain = detectDomain(message);
    const requestedSchemaName = extractRequestedSchemaName(message);
    const requestedEndpoint = extractRequestedEndpoint(message);

    if (domain && requestedSchemaName) {
      const result = await buildSchemaDetailResponse(domain, requestedSchemaName, audience);
      return res.json({
        ...result,
        tool_name: 'schema_detail_pipeline',
        tool_arguments: { domain_name: domain, schema_name: requestedSchemaName, audience },
        explore_next: [
          `Show me the schemas of the ${domain} API`,
          `Show example endpoints for the ${domain} API`,
          `Show me the business capabilities of the ${domain} API`
        ]
      });
    }

    if (domain && wantsSubApiDesign(message)) {
      const result = await buildSubApiResponse(domain, message, audience);
      return res.json({
        ...result,
        audience,
        tool_name: 'sub_api_pipeline',
        tool_arguments: { domain_name: domain, audience },
        explore_next: [
          `List operations for the ${domain} API`,
          `Show me the schemas of the ${domain} API`,
          `Show example endpoints for the ${domain} API`
        ]
      });
    }

    if (domain && requestedEndpoint) {
      const result = await buildEndpointDetailResponse(domain, requestedEndpoint, audience);
      return res.json({
        ...result,
        tool_name: 'endpoint_detail_pipeline',
        tool_arguments: { domain_name: domain, endpoint: requestedEndpoint, audience },
        explore_next: [
          `Show me the schemas of the ${domain} API`,
          `Show me the business capabilities of the ${domain} API`
        ]
      });
    }

    if (domain && wantsRichApiDescription(message)) {
      const result = await buildRichApiDescriptionResponse(domain, message, audience);
      return res.json({
        ...result,
        audience,
        tool_name: 'rich_api_description_pipeline',
        tool_arguments: { domain_name: domain, audience },
        explore_next: [
          `Show me the business capabilities of the ${domain} API`,
          `Show me the schemas of the ${domain} API`,
          `Show example endpoints for the ${domain} API`
        ]
      });
    }

    if (domain && wantsCapabilities(message)) {
      const result = await buildCapabilityResponse(domain, audience);
      return res.json({
        ...result,
        audience,
        tool_name: 'business_capability_navigator',
        tool_arguments: { domain_name: domain, audience },
        explore_next: [
          `List operations for the ${domain} API`,
          `List schemas for the ${domain} API`,
          `Show example endpoints for the ${domain} API`,
          `Show more schemas for the ${domain} API`
        ]
      });
    }

    if (domain && wantsOverview(message)) {
      const result = await buildOverviewResponse(domain, message, audience);
      return res.json({
        ...result,
        audience,
        tool_name: 'overview_pipeline',
        tool_arguments: { domain_name: domain, audience },
        explore_next: [
          `Show me the business capabilities of the ${domain} API`,
          `List schemas for the ${domain} API`,
          `Show example endpoints for the ${domain} API`,
          `Show more schemas for the ${domain} API`
        ]
      });
    }

    if (domain && wantsSchemas(message)) {
      const result = await buildSchemaResponse(domain, message, audience);
      return res.json({
        ...result,
        audience,
        tool_name: 'schema_pipeline',
        tool_arguments: { domain_name: domain, audience },
        explore_next: [
          `Show more schemas for the ${domain} API`,
          `Show me the business capabilities of the ${domain} API`,
          `Show example endpoints for the ${domain} API`
        ]
      });
    }

    if (domain && wantsEndpoints(message)) {
      const result = await buildEndpointResponse(domain, message, audience);
      return res.json({
        ...result,
        audience,
        tool_name: 'endpoint_pipeline',
        tool_arguments: { domain_name: domain, audience },
        explore_next: [
          `Show me the schemas of the ${domain} API`,
          `Show me the business capabilities of the ${domain} API`
        ]
      });
    }

    const result = await toolCached('listDomains', {});
    const answer = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: `User:
${message}

Audience:
${audience}

Data:
${JSON.stringify(result)}

Answer clearly and concisely in the context of STAR automotive APIs.
If the user is asking to design, derive, propose, build, or outline a sub-API, provide a structured proposal with:
1. Direct answer
2. Proposed resources or operations
3. Suggested next steps`
      }),
      15000,
      'OpenAI answer generation'
    );

    return res.json({
      answer: answer.output_text,
      sections: extractSections(answer.output_text),
      audience,
      capability_cards: [],
      endpoint_cards: [],
      schema_cards: [],
      workflow_map: [],
      explore_next: [],
      tool_name: 'listDomains',
      tool_arguments: { audience }
    });
  } catch (e) {
    const msg = e?.message || 'Unknown error';

    if (isRateLimitErrorMessage(msg)) {
      return res.status(429).json({
        error: 'The service is being rate limited right now. Please wait a few seconds and try again.'
      });
    }

    if (msg.includes('timed out')) {
      return res.status(504).json({ error: msg });
    }

    return res.status(500).json({ error: msg });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`STAR audience-aware server running on ${PORT}`);
});
