import 'dotenv/config';
import express from 'express';
import OpenAI from 'openai';
import fs from 'node:fs';
import path from 'node:path';
import yaml from 'js-yaml';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const MODEL = process.env.OPENAI_MODEL || 'gpt-5';

const app = express();
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const client = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

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
 * DOMAIN -> OPENAPI SPEC REGISTRY
 * -------------------------------------------------- */

const DOMAIN_SPEC_PATHS = {
  appointment:
    process.env.APPOINTMENT_OPENAPI_SPEC_PATH || './openapi/appointment-api.yaml',
  'multi-point-inspection':
    process.env.MPI_OPENAPI_SPEC_PATH || './openapi/multi-point-inspection-api.yaml'
};

function loadYamlFile(filePath) {
  try {
    const resolved = path.resolve(__dirname, filePath);

    if (!fs.existsSync(resolved)) {
      return { spec: null, sourcePath: resolved, error: 'File not found' };
    }

    const raw = fs.readFileSync(resolved, 'utf8');
    const spec = yaml.load(raw);

    if (!spec || typeof spec !== 'object') {
      return { spec: null, sourcePath: resolved, error: 'Invalid YAML object' };
    }

    return { spec, sourcePath: resolved, error: null };
  } catch (err) {
    return { spec: null, sourcePath: filePath, error: err.message };
  }
}

function loadOpenApiRegistry() {
  const registry = {};

  for (const [domain, filePath] of Object.entries(DOMAIN_SPEC_PATHS)) {
    const loaded = loadYamlFile(filePath);
    registry[domain] = loaded;

    if (loaded.spec) {
      console.log(`Loaded OpenAPI spec for ${domain} from ${loaded.sourcePath}`);
    } else {
      console.warn(
        `Failed loading OpenAPI spec for ${domain} from ${loaded.sourcePath}: ${loaded.error}`
      );
    }
  }

  return registry;
}

const OPENAPI_REGISTRY = loadOpenApiRegistry();

function getOpenApiSpec(domain) {
  return OPENAPI_REGISTRY[domain]?.spec || null;
}

function getOpenApiSourcePath(domain) {
  return OPENAPI_REGISTRY[domain]?.sourcePath || null;
}

function getOpenApiSchemaMap(domain) {
  return getOpenApiSpec(domain)?.components?.schemas || {};
}

/* --------------------------------------------------
 * GENERAL HELPERS
 * -------------------------------------------------- */

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

function normalizeAudience(audience = '') {
  const a = String(audience).toLowerCase();
  if (a === 'technical') return 'technical';
  if (a === 'architecture') return 'architecture';
  return 'business';
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
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeEndpointTitle(endpointTitle = '') {
  return canonicalizeEndpointAlias(normalizeEndpointText(endpointTitle));
}

function normalizePathForMatch(routePath = '') {
  return canonicalizeEndpointAlias(String(routePath).toUpperCase())
    .replace(/\{[^}]+\}/g, '{ID}')
    .replace(/\/+$/, '');
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

  return null;
}

function wantsCapabilities(message = '') {
  const m = normalizeLooseText(message);
  return (
    m.includes('capabilities') ||
    m.includes('business capabilities') ||
    m.includes('service advisor') ||
    m.includes('service lane') ||
    m.includes('advisor-facing')
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
    m.includes('describe the api') ||
    m.includes('rich description') ||
    m.includes('detailed description')
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
    m.includes('build a sub-api') ||
    m.includes('propose a sub-api')
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
    if (match?.[1]) return match[1].trim();
  }

  return null;
}

function extractRequestedEndpoint(message = '') {
  const text = String(message || '').trim();
  const match = text.match(/(GET|POST|PUT|PATCH|DELETE)\s+\/[A-Za-z0-9/_{}-]+/i);
  if (!match) return null;
  return normalizeEndpointTitle(match[0]);
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

/* --------------------------------------------------
 * OPENAPI EXTRACTION
 * -------------------------------------------------- */

function schemaNameFromRef(ref = '') {
  const match = String(ref).match(/#\/components\/schemas\/(.+)$/);
  return match?.[1] || null;
}

function dereferenceSchemaRef(domain, ref) {
  const name = schemaNameFromRef(ref);
  if (!name) return null;
  const raw = getOpenApiSchemaMap(domain)[name];
  if (!raw) return null;
  return { name, raw };
}

function buildSampleFromSchema(domain, schema, depth = 0) {
  if (!schema || depth > 4) return null;

  if (schema.example !== undefined) return schema.example;

  if (schema.$ref) {
    const deref = dereferenceSchemaRef(domain, schema.$ref);
    return deref ? buildSampleFromSchema(domain, deref.raw, depth + 1) : null;
  }

  if (schema.enum?.length) return schema.enum[0];

  if (schema.type === 'object' || schema.properties) {
    const out = {};
    for (const [key, value] of Object.entries(schema.properties || {})) {
      const sample = buildSampleFromSchema(domain, value, depth + 1);
      out[key] = sample !== null ? sample : '<value>';
    }
    return out;
  }

  if (schema.type === 'array') {
    const item = buildSampleFromSchema(domain, schema.items, depth + 1);
    return [item !== null ? item : '<item>'];
  }

  switch (schema.type) {
    case 'string':
      if (schema.format === 'date-time') return '2026-03-25T12:00:00Z';
      if (schema.format === 'date') return '2026-03-25';
      if (schema.format === 'uri') return 'https://example.com/resource';
      return '<string>';
    case 'integer':
    case 'number':
      return 0;
    case 'boolean':
      return true;
    default:
      return null;
  }
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

function collectSchemaNamesFromRefs(value, acc = new Set()) {
  if (!value || typeof value !== 'object') return acc;

  if (typeof value.$ref === 'string') {
    const name = schemaNameFromRef(value.$ref);
    if (name) acc.add(name);
  }

  for (const child of Object.values(value)) {
    if (child && typeof child === 'object') {
      collectSchemaNamesFromRefs(child, acc);
    }
  }

  return acc;
}

function getOpenApiOperations(domain) {
  const spec = getOpenApiSpec(domain);
  if (!spec?.paths) return [];

  const out = [];
  for (const [routePath, pathItem] of Object.entries(spec.paths)) {
    for (const method of ['get', 'post', 'put', 'patch', 'delete']) {
      const op = pathItem?.[method];
      if (!op) continue;

      out.push({
        method: method.toUpperCase(),
        path: routePath,
        summary: op.summary || '',
        description: op.description || '',
        operationId: op.operationId || '',
        tags: Array.isArray(op.tags) ? op.tags : [],
        raw: op,
        pathItem
      });
    }
  }
  return out;
}

function getOpenApiOperationsForDomain(domain) {
  return getOpenApiOperations(domain);
}

function getOpenApiSchemasForDomain(domain) {
  const schemaMap = getOpenApiSchemaMap(domain);
  const names = Object.keys(schemaMap);
  if (!names.length) return [];

  return names.map((name) => ({
    name,
    description: schemaMap[name]?.description || '',
    raw: schemaMap[name]
  }));
}

function findOpenApiSchemaByName(domain, schemaName) {
  const schemaMap = getOpenApiSchemaMap(domain);
  const target = String(schemaName || '').trim().toLowerCase();
  if (!target) return null;

  for (const [name, raw] of Object.entries(schemaMap)) {
    if (name.toLowerCase() === target) return { name, raw };
  }

  for (const [name, raw] of Object.entries(schemaMap)) {
    if (name.toLowerCase().includes(target) || target.includes(name.toLowerCase())) {
      return { name, raw };
    }
  }

  return null;
}

function findOpenApiEndpoint(domain, endpointTitle) {
  const spec = getOpenApiSpec(domain);
  if (!spec?.paths) return null;

  const normalizedTarget = normalizeEndpointTitle(endpointTitle);
  const spaceIndex = normalizedTarget.indexOf(' ');
  if (spaceIndex === -1) return null;

  const method = normalizedTarget.slice(0, spaceIndex);
  const routePath = normalizedTarget.slice(spaceIndex + 1);
  const targetNormalized = normalizePathForMatch(routePath);

  for (const [pathKey, pathItem] of Object.entries(spec.paths)) {
    const candidateNormalized = normalizePathForMatch(pathKey);
    if (candidateNormalized !== targetNormalized) continue;

    const op = pathItem?.[method.toLowerCase()];
    if (!op) continue;

    return {
      method,
      path: pathKey,
      details: op,
      pathItem
    };
  }

  return null;
}

function extractParametersFromOperation(operation, pathItem = {}) {
  const allParams = [
    ...(Array.isArray(pathItem.parameters) ? pathItem.parameters : []),
    ...(Array.isArray(operation.parameters) ? operation.parameters : [])
  ];

  const deduped = [];
  const seen = new Set();

  for (const p of allParams) {
    const key = `${p?.name || ''}:${p?.in || ''}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push({
      name: p?.name || '',
      in: p?.in || '',
      required: Boolean(p?.required),
      description: p?.description || '',
      schema: p?.schema || null,
      example: p?.example ?? p?.schema?.example ?? null
    });
  }

  return deduped;
}

function extractRelatedSchemasFromOperation(domain, op) {
  const found = collectSchemaNamesFromRefs(op, new Set());
  return Array.from(found).map((name) => ({
    title: name,
    description: getOpenApiSchemaMap(domain)[name]?.description || 'Related OpenAPI schema',
    prompt: `Show schema ${name}`
  }));
}

function extractResponseSchemaCardFromOperation(domain, op) {
  const responses = op?.responses || {};
  const preferred = responses['200'] || responses['201'] || responses['202'] || Object.values(responses)[0];
  if (!preferred) return [];

  const content = findPrimaryJsonContent(preferred.content);
  if (!content?.schema) return [];

  const found = collectSchemaNamesFromRefs(content.schema, new Set());
  return Array.from(found).map((name) => ({
    title: name,
    description: getOpenApiSchemaMap(domain)[name]?.description || 'Response schema',
    prompt: `Show schema ${name}`
  }));
}

function extractRequestExampleFromOperation(domain, op) {
  const content = findPrimaryJsonContent(op?.requestBody?.content);
  if (!content) return null;

  if (content.example !== undefined) return content.example;
  if (content.examples && typeof content.examples === 'object') {
    const first = Object.values(content.examples)[0];
    if (first?.value !== undefined) return first.value;
  }

  return buildSampleFromSchema(domain, content.schema);
}

function extractResponseExampleFromOperation(domain, op) {
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

  return buildSampleFromSchema(domain, content.schema);
}

/* --------------------------------------------------
 * BUILT-IN FALLBACKS
 * -------------------------------------------------- */

const DOMAIN_OVERVIEWS = {
  appointment: {
    business: `1. Direct answer
The Appointment API represents the dealership service scheduling and intake layer.

2. Core responsibilities
It manages booking context, requested services, vehicle and customer references, and appointment lifecycle.

3. Useful next steps
Explore endpoints, schemas, and service-lane workflows.`,
    technical: `1. Direct answer
The Appointment API is the scheduling and intake domain for dealership service operations.

2. Core responsibilities
It models appointment resources, requested services, time context, and upstream references.

3. Useful next steps
Inspect schemas, endpoints, and reference relationships.`,
    architecture: `1. Direct answer
The Appointment API owns scheduling and pre-intake coordination.

2. Core responsibilities
It establishes service context that downstream domains consume.

3. Useful next steps
Review domain boundaries, handoffs, and integration points.`
  },
  'multi-point-inspection': {
    business: `1. Direct answer
The Multi-Point Inspection API supports structured vehicle health checks in the service lane.

2. Core responsibilities
It covers inspection lifecycle, findings, recommendations, approvals, and communication flow.

3. Useful next steps
Explore business capabilities, endpoints, schemas, and workflow stages.`,
    technical: `1. Direct answer
The Multi-Point Inspection API models inspection lifecycle, findings, media, recommendations, and outcomes.

2. Core responsibilities
It provides the domain structures and operations needed to create, track, review, and complete inspections.

3. Useful next steps
Inspect endpoint definitions, schema structures, and related object families.`,
    architecture: `1. Direct answer
The Multi-Point Inspection API owns structured inspection and recommendation workflow.

2. Core responsibilities
It bridges inspection observations to advisor and execution decisions.

3. Useful next steps
Review cross-domain relationships and downstream handoffs.`
  }
};

function getAudienceOverview(domain, audience) {
  const map = DOMAIN_OVERVIEWS[domain];
  if (!map) return 'No overview available.';
  if (audience === 'technical') return map.technical;
  if (audience === 'architecture') return map.architecture;
  return map.business;
}

function getBuiltInCapabilityCards(domain) {
  if (domain === 'multi-point-inspection') {
    return [
      {
        title: 'Start inspection workflow',
        description: 'Begin the MPI process from intake, appointment context, or repair-order workflow.',
        prompt: 'Explain how to start the inspection workflow in the multi-point-inspection API'
      },
      {
        title: 'Review findings and media',
        description: 'Review technician findings, notes, photos, and condition evidence.',
        prompt: 'Explain how to review findings and media in the multi-point-inspection API'
      },
      {
        title: 'Capture customer approval',
        description: 'Record customer decisions, approvals, and communication outcomes.',
        prompt: 'Explain how customer approval works in the multi-point-inspection API'
      }
    ];
  }

  if (domain === 'appointment') {
    return [
      {
        title: 'Schedule appointment',
        description: 'Create and manage service appointments for customers and vehicles.',
        prompt: 'Explain how to schedule and manage appointments in the Appointment API'
      },
      {
        title: 'Confirm service needs',
        description: 'Validate requested services, timing, and customer expectations.',
        prompt: 'Explain how a service advisor confirms service needs using the Appointment API'
      }
    ];
  }

  return [];
}

function getBuiltInWorkflowMap(domain) {
  if (domain === 'multi-point-inspection') {
    return [
      { step: 'Start inspection', detail: 'Launch MPI workflow tied to intake or RO context', prompt: 'Explain how to start the inspection workflow in the multi-point-inspection API' },
      { step: 'Capture findings', detail: 'Technician records results, notes, and media', prompt: 'Explain how findings are captured in the multi-point-inspection API' },
      { step: 'Build recommendations', detail: 'Advisor prepares customer-facing repair guidance', prompt: 'Explain how recommendations are built in the multi-point-inspection API' },
      { step: 'Get approval', detail: 'Customer decisions are captured and tracked', prompt: 'Explain customer approval flow in the multi-point-inspection API' }
    ];
  }

  if (domain === 'appointment') {
    return [
      { step: 'Schedule appointment', detail: 'Customer booking and advisor intake', prompt: 'Explain the scheduling flow of the Appointment API' },
      { step: 'Confirm service needs', detail: 'Advisor validates request and timing', prompt: 'Explain advisor validation workflow in the Appointment API' },
      { step: 'Prepare service lane', detail: 'Appointment data supports dealer operations', prompt: 'Explain how the Appointment API supports service-lane preparation' }
    ];
  }

  return [];
}

function getBuiltInSchemaCards(domain) {
  if (domain === 'multi-point-inspection') {
    return [
      { title: 'Inspection', description: 'Core inspection entity representing the inspection lifecycle.', prompt: 'Show schema Inspection in the multi-point-inspection API' },
      { title: 'Finding', description: 'Represents an inspection finding, condition, or observed result.', prompt: 'Show schema Finding in the multi-point-inspection API' },
      { title: 'Recommendation', description: 'Represents advisor-ready or technician-derived recommended work.', prompt: 'Show schema Recommendation in the multi-point-inspection API' },
      { title: 'Approval', description: 'Captures customer approval or decline decisions.', prompt: 'Show schema Approval in the multi-point-inspection API' },
      { title: 'InspectionMedia', description: 'Represents photos, videos, or other evidence attached to findings.', prompt: 'Show schema InspectionMedia in the multi-point-inspection API' }
    ];
  }

  if (domain === 'appointment') {
    return [
      { title: 'Appointment', description: 'Core appointment entity for dealership service scheduling.', prompt: 'Show schema Appointment in the Appointment API' },
      { title: 'RequestedService', description: 'Represents requested service work tied to an appointment.', prompt: 'Show schema RequestedService in the Appointment API' },
      { title: 'AppointmentStatus', description: 'Represents appointment lifecycle or state.', prompt: 'Show schema AppointmentStatus in the Appointment API' }
    ];
  }

  return [];
}

function getBuiltInEndpointCards(domain) {
  if (domain === 'multi-point-inspection') {
    return [
      { title: 'GET /inspections', description: 'List or search inspections.', prompt: 'Explain GET /inspections for the multi-point-inspection API' },
      { title: 'POST /inspections', description: 'Create or start an inspection.', prompt: 'Explain POST /inspections for the multi-point-inspection API' },
      { title: 'GET /inspections/{id}', description: 'Retrieve a specific inspection.', prompt: 'Explain GET /inspections/{id} for the multi-point-inspection API' },
      { title: 'GET /inspections/{id}/findings', description: 'Retrieve findings tied to an inspection.', prompt: 'Explain GET /inspections/{id}/findings for the multi-point-inspection API' }
    ];
  }

  if (domain === 'appointment') {
    return [
      { title: 'GET /appointments', description: 'List or search appointments.', prompt: 'Explain GET /appointments for the Appointment API' },
      { title: 'POST /appointments', description: 'Create a new appointment.', prompt: 'Explain POST /appointments for the Appointment API' },
      { title: 'GET /appointments/{id}', description: 'Retrieve a specific appointment.', prompt: 'Explain GET /appointments/{id} for the Appointment API' }
    ];
  }

  return [];
}

function getBuiltInEndpointDetail(domain, endpointTitle) {
  const key = normalizeEndpointTitle(endpointTitle);

  const builtIn = {
    appointment: {
      'GET /APPOINTMENTS': `1. Direct answer
GET /appointments lists or searches appointments.

2. Key endpoint details
This endpoint supports schedule views, search flows, and operational appointment lookup.

3. Why it matters
It is the main collection endpoint for the Appointment API.`,
      'GET /APPOINTMENTS/{ID}': `1. Direct answer
GET /appointments/{id} retrieves a specific appointment.

2. Key endpoint details
This endpoint returns the state and details of one appointment instance.

3. Why it matters
It is the primary read path for single-appointment lookup.`
    },
    'multi-point-inspection': {
      'GET /INSPECTIONS': `1. Direct answer
GET /inspections lists or searches inspection records.

2. Key endpoint details
This endpoint is useful for retrieving inspection collections, filtering workflow state, and supporting advisor or operational views of inspection activity.

3. Why it matters
It gives technical consumers a collection-level entry point into the MPI domain.`,
      'GET /INSPECTIONS/{ID}': `1. Direct answer
GET /inspections/{id} retrieves a specific inspection.

2. Key endpoint details
This endpoint is used to inspect the full state of a single inspection, including status, relationships, and downstream context.

3. Why it matters
It is the most direct technical path for reading a specific MPI entity.`,
      'GET /INSPECTIONS/{ID}/FINDINGS': `1. Direct answer
GET /inspections/{id}/findings retrieves findings for a specific inspection.

2. Key endpoint details
This endpoint returns the technician-recorded findings associated with the inspection, often including condition results, notes, severity or status, and related media references.

3. Why it matters
It is the main read path for inspection results and supports advisor review, customer communication, and recommendation workflows.`
    }
  };

  return builtIn[domain]?.[key] || null;
}

/* --------------------------------------------------
 * RESPONSE BUILDERS
 * -------------------------------------------------- */

async function buildOverviewResponse(domain, message, audience) {
  const overviewText = getAudienceOverview(domain, audience);
  let answer = overviewText;

  if (client) {
    try {
      const response = await withTimeout(
        client.responses.create({
          model: MODEL,
          input: `Audience: ${audience}
Domain: ${domain}
User request: ${message}

Base overview:
${overviewText}

Write a concise but rich structured response.`
        }),
        9000,
        'OpenAI overview generation'
      );
      answer = response.output_text || overviewText;
    } catch {
      answer = overviewText;
    }
  }

  const ops = getOpenApiOperationsForDomain(domain);
  const schemas = getOpenApiSchemasForDomain(domain);

  return {
    answer,
    sections: extractSections(answer),
    audience,
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: ops.length
      ? ops.slice(0, 8).map((op) => ({
          title: `${op.method} ${op.path}`,
          description: op.summary || op.description || 'API endpoint',
          prompt: `Explain ${op.method} ${op.path} for the ${domain} API`
        }))
      : getBuiltInEndpointCards(domain),
    schema_cards: schemas.length
      ? schemas.slice(0, 12).map((s) => ({
          title: s.name,
          description: s.description || 'View schema details',
          prompt: `Show schema ${s.name} in the ${domain} API`
        }))
      : getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: !ops.length || !schemas.length,
    source: ops.length || schemas.length ? 'openapi' : 'fallback'
  };
}

async function buildCapabilitiesResponse(domain, audience) {
  const answer = audience === 'technical'
    ? `1. Direct answer
Technical capability views for the ${domain} API are shown below.

2. Key details
These capabilities align workflow meaning with technical structures.

3. Useful next steps
Select a capability card to explore schemas, endpoints, and workflow relationships.`
    : audience === 'architecture'
      ? `1. Direct answer
Architecture-oriented capability views for the ${domain} API are shown below.

2. Key details
These capabilities highlight responsibilities, handoffs, and model boundaries.

3. Useful next steps
Select a capability card to explore integration implications.`
      : `1. Direct answer
Business capabilities for the ${domain} API are shown below.

2. Key details
These capabilities align with dealership workflow and service-lane usage.

3. Useful next steps
Select a capability card to explore the domain in more detail.`;

  const ops = getOpenApiOperationsForDomain(domain);
  const schemas = getOpenApiSchemasForDomain(domain);

  return {
    answer,
    sections: extractSections(answer),
    audience,
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: ops.length
      ? ops.slice(0, 8).map((op) => ({
          title: `${op.method} ${op.path}`,
          description: op.summary || op.description || 'API endpoint',
          prompt: `Explain ${op.method} ${op.path} for the ${domain} API`
        }))
      : getBuiltInEndpointCards(domain),
    schema_cards: schemas.length
      ? schemas.slice(0, 12).map((s) => ({
          title: s.name,
          description: s.description || 'View schema details',
          prompt: `Show schema ${s.name} in the ${domain} API`
        }))
      : getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: !ops.length || !schemas.length,
    source: ops.length || schemas.length ? 'openapi' : 'fallback'
  };
}

async function buildSubApiResponse(domain, message, audience) {
  const ops = getOpenApiOperationsForDomain(domain);
  const getOps = ops.filter((op) => op.method === 'GET');

  let answer = `1. Direct answer
A focused sub-API can be derived for the ${domain} domain.

2. Proposed resources or endpoints
Start with a reduced set of read-oriented GET operations that expose the main collection and entity retrieval flows.

3. Proposed schemas
Reuse the core read models and supporting reference schemas needed for consumers.

4. Design notes
Prefer a narrow read-only surface, stable identifiers, and minimal cross-domain coupling.

5. Suggested next steps
Select the GET endpoints you want to keep and then trim the schema surface to only what those responses need.`;

  if (client) {
    try {
      const response = await withTimeout(
        client.responses.create({
          model: MODEL,
          input: `Audience: ${audience}
Domain: ${domain}
User request: ${message}

Available GET operations:
${JSON.stringify(getOps.map((op) => ({
  method: op.method,
  path: op.path,
  summary: op.summary,
  description: op.description
})), null, 2)}

Write a structured sub-API proposal.`
        }),
        12000,
        'OpenAI sub-api generation'
      );
      answer = response.output_text || answer;
    } catch {}
  }

  return {
    answer,
    sections: extractSections(answer),
    audience,
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: getOps.length
      ? getOps.map((op) => ({
          title: `${op.method} ${op.path}`,
          description: op.summary || op.description || 'GET endpoint',
          prompt: `Explain ${op.method} ${op.path} for the ${domain} API`
        }))
      : getBuiltInEndpointCards(domain),
    schema_cards: getOpenApiSchemasForDomain(domain).slice(0, 12).map((s) => ({
      title: s.name,
      description: s.description || 'Schema',
      prompt: `Show schema ${s.name} in the ${domain} API`
    })),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: false,
    source: getOps.length ? 'openapi' : 'fallback'
  };
}

async function buildSchemaListResponse(domain, audience) {
  const schemas = getOpenApiSchemasForDomain(domain);

  const answer = `1. Direct answer
Representative schemas for the ${domain} API are shown below.

2. Key schema details
These schema cards represent the primary object families and data structures users typically need first.

3. Useful next steps
Open a schema card to view its OpenAPI structure and related schema relationships.`;

  return {
    answer,
    sections: extractSections(answer),
    audience,
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: [],
    schema_cards: schemas.length
      ? schemas.map((s) => ({
          title: s.name,
          description: s.description || 'View schema details',
          prompt: `Show schema ${s.name} in the ${domain} API`
        }))
      : getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: !schemas.length,
    source: schemas.length ? 'openapi' : 'fallback'
  };
}

async function buildSchemaDetailResponse(domain, schemaName, audience) {
  const found = findOpenApiSchemaByName(domain, schemaName);

  if (found) {
    const answer = client
      ? await (async () => {
          try {
            const response = await withTimeout(
              client.responses.create({
                model: MODEL,
                input: `Audience: ${audience}
Schema name: ${found.name}
Schema data:
${JSON.stringify(found.raw, null, 2)}

Write:
1. Direct answer
2. Key schema details
3. Why it matters
4. Useful next steps`
              }),
              9000,
              'OpenAI schema detail generation'
            );
            return response.output_text;
          } catch {
            return `1. Direct answer
${found.name} is a schema in the ${domain} API.

2. Key schema details
This schema is available from the OpenAPI definition and can be inspected in raw form.

3. Why it matters
It represents part of the structural contract of the domain.

4. Useful next steps
Review related schemas and endpoints that reference it.`;
          }
        })()
      : `1. Direct answer
${found.name} is a schema in the ${domain} API.

2. Key schema details
This schema is available from the OpenAPI definition and can be inspected in raw form.

3. Why it matters
It represents part of the structural contract of the domain.

4. Useful next steps
Review related schemas and endpoints that reference it.`;

    const related = Array.from(collectSchemaNamesFromRefs(found.raw, new Set())).map((name) => ({
      title: name,
      description: getOpenApiSchemaMap(domain)[name]?.description || 'Related schema',
      prompt: `Show schema ${name}`
    }));

    return {
      answer,
      sections: extractSections(answer),
      audience,
      schema_title: found.name,
      raw_schema: found.raw,
      raw_schema_format: 'json',
      related_schema_cards: related,
      capability_cards: getBuiltInCapabilityCards(domain),
      endpoint_cards: getBuiltInEndpointCards(domain),
      schema_cards: getBuiltInSchemaCards(domain),
      workflow_map: getBuiltInWorkflowMap(domain),
      progressive: false,
      source: 'openapi'
    };
  }

  const fallbackAnswer = getBuiltInSchemaCards(domain).length
    ? `1. Direct answer
${schemaName} is a schema in the ${domain} API.

2. Key schema details
Detailed OpenAPI schema content is not available right now.

3. Useful next steps
Inspect related schema cards or try a different schema name.`
    : 'Schema not found.';

  return {
    answer: fallbackAnswer,
    sections: extractSections(fallbackAnswer),
    audience,
    schema_title: schemaName,
    raw_schema: null,
    raw_schema_format: 'json',
    related_schema_cards: [],
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: getBuiltInEndpointCards(domain),
    schema_cards: getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: true,
    source: 'fallback'
  };
}

async function buildEndpointListResponse(domain, audience) {
  const ops = getOpenApiOperationsForDomain(domain);
  const answer = `1. Direct answer
Representative endpoints for the ${domain} API are shown below.

2. Key endpoint details
These endpoints represent the main operation families for the domain.

3. Useful next steps
Select an endpoint to inspect its parameters, examples, schemas, and raw OpenAPI structure.`;

  return {
    answer,
    sections: extractSections(answer),
    audience,
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: ops.length
      ? ops.map((op) => ({
          title: `${op.method} ${op.path}`,
          description: op.summary || op.description || 'API endpoint',
          prompt: `Explain ${op.method} ${op.path} for the ${domain} API`
        }))
      : getBuiltInEndpointCards(domain),
    schema_cards: getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: !ops.length,
    source: ops.length ? 'openapi' : 'fallback'
  };
}

async function buildEndpointDetailResponse(domain, endpointTitle, audience) {
  const openApiEndpoint = findOpenApiEndpoint(domain, endpointTitle);
  const builtIn = getBuiltInEndpointDetail(domain, endpointTitle);

  if (openApiEndpoint) {
    const op = openApiEndpoint.details;
    const parameters = extractParametersFromOperation(op, openApiEndpoint.pathItem);
    const relatedSchemaCards = [
      ...extractRelatedSchemasFromOperation(domain, op),
      ...extractResponseSchemaCardFromOperation(domain, op)
    ].filter((item, idx, arr) => arr.findIndex((x) => x.title === item.title) === idx);

    const requestExample = extractRequestExampleFromOperation(domain, op);
    const responseExample = extractResponseExampleFromOperation(domain, op);

    const answer = client
      ? await (async () => {
          try {
            const response = await withTimeout(
              client.responses.create({
                model: MODEL,
                input: `Audience: ${audience}
Endpoint: ${openApiEndpoint.method} ${openApiEndpoint.path}

Summary: ${op.summary || ''}
Description: ${op.description || ''}
Parameters:
${JSON.stringify(parameters, null, 2)}

Write:
1. Direct answer
2. Key endpoint details
3. Why it matters
4. Useful next steps`
              }),
              9000,
              'OpenAI endpoint detail generation'
            );
            return response.output_text;
          } catch {
            return `1. Direct answer
${openApiEndpoint.method} ${openApiEndpoint.path} is a defined endpoint in the ${domain} API.

2. Key endpoint details
${op.summary || op.description || 'This endpoint is part of the domain API surface.'}

3. Why it matters
This endpoint is relevant to consumers who need direct access to this resource or workflow step.

4. Useful next steps
Review the parameters, related schemas, examples, and raw OpenAPI definition.`;
          }
        })()
      : `1. Direct answer
${openApiEndpoint.method} ${openApiEndpoint.path} is a defined endpoint in the ${domain} API.

2. Key endpoint details
${op.summary || op.description || 'This endpoint is part of the domain API surface.'}

3. Why it matters
This endpoint is relevant to consumers who need direct access to this resource or workflow step.

4. Useful next steps
Review the parameters, related schemas, examples, and raw OpenAPI definition.`;

    return {
      answer,
      sections: extractSections(answer),
      audience,
      endpoint_title: `${openApiEndpoint.method} ${openApiEndpoint.path}`,
      endpoint_parameters: parameters,
      related_schema_cards: relatedSchemaCards,
      request_example: requestExample,
      response_example: responseExample,
      raw_openapi: op,
      capability_cards: getBuiltInCapabilityCards(domain),
      endpoint_cards: getBuiltInEndpointCards(domain),
      schema_cards: getBuiltInSchemaCards(domain),
      workflow_map: getBuiltInWorkflowMap(domain),
      progressive: false,
      source: 'openapi'
    };
  }

  const key = normalizeEndpointTitle(endpointTitle);
  let parameters = [];
  let relatedSchemaCards = [];
  let requestExample = null;
  let responseExample = null;

  if (domain === 'multi-point-inspection' && key === 'GET /INSPECTIONS/{ID}/FINDINGS') {
    parameters = [
      {
        name: 'id',
        in: 'path',
        required: true,
        description: 'Unique identifier of the inspection whose findings should be retrieved.'
      }
    ];
    relatedSchemaCards = [
      { title: 'Inspection', description: 'Parent inspection entity for the findings collection.', prompt: 'Show schema Inspection in the multi-point-inspection API' },
      { title: 'Finding', description: 'Represents an inspection finding, observed condition, or result.', prompt: 'Show schema Finding in the multi-point-inspection API' },
      { title: 'InspectionMedia', description: 'Represents photos, videos, or evidence attached to findings.', prompt: 'Show schema InspectionMedia in the multi-point-inspection API' }
    ];
    requestExample = {
      method: 'GET',
      path: '/inspections/inspection-123/findings'
    };
    responseExample = {
      findings: [
        {
          finding_id: 'finding-001',
          inspection_id: 'inspection-123',
          status: 'yellow',
          condition: 'Brake pads worn',
          notes: 'Front brake pads below recommended thickness.',
          media: [
            {
              media_id: 'media-01',
              media_type: 'image',
              uri: 'https://example.com/media/brakes-front-left.jpg'
            }
          ]
        }
      ]
    };
  }

  const answer = builtIn || `1. Direct answer
${endpointTitle} is an endpoint in the ${domain} API.

2. Key endpoint details
Detailed live endpoint content is not available right now.

3. Useful next steps
Inspect related schemas or try again later.`;

  return {
    answer,
    sections: extractSections(answer),
    audience,
    endpoint_title: normalizeEndpointTitle(endpointTitle),
    endpoint_parameters: parameters,
    related_schema_cards: relatedSchemaCards,
    request_example: requestExample,
    response_example: responseExample,
    raw_openapi: null,
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: getBuiltInEndpointCards(domain),
    schema_cards: getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: true,
    source: 'fallback'
  };
}

/* --------------------------------------------------
 * OPENAPI EXPLORER ROUTES
 * -------------------------------------------------- */

app.get('/api/version', (_req, res) => {
  res.json({
    version: 'multi-spec-registry',
    domains: Object.fromEntries(
      Object.keys(DOMAIN_SPEC_PATHS).map((domain) => [
        domain,
        {
          openapi_loaded: Boolean(getOpenApiSpec(domain)),
          openapi_source: getOpenApiSourcePath(domain)
        }
      ])
    )
  });
});

app.get('/api/openapi-index', (req, res) => {
  const domain = String(req.query.domain || '').trim();
  if (!domain) {
    return res.status(400).json({ error: 'domain is required.' });
  }

  const spec = getOpenApiSpec(domain);
  if (!spec?.paths) {
    return res.status(404).json({ error: `OpenAPI spec not loaded for domain: ${domain}` });
  }

  const entries = getOpenApiOperations(domain).map((op) => ({
    method: op.method,
    path: op.path,
    summary: op.summary,
    description: op.description,
    operationId: op.operationId,
    tags: op.tags
  }));

  return res.json({
    domain,
    title: spec.info?.title || 'OpenAPI',
    version: spec.info?.version || '',
    source_path: getOpenApiSourcePath(domain),
    operations: entries
  });
});

app.get('/api/openapi-endpoint', (req, res) => {
  const domain = String(req.query.domain || '').trim();
  const method = String(req.query.method || '').toUpperCase();
  const routePath = String(req.query.path || '');

  if (!domain || !method || !routePath) {
    return res.status(400).json({ error: 'domain, method, and path are required.' });
  }

  const found = findOpenApiEndpoint(domain, `${method} ${routePath}`);
  if (!found) {
    return res.status(404).json({ error: `Endpoint not found in OpenAPI for domain: ${domain}` });
  }

  const parameters = extractParametersFromOperation(found.details, found.pathItem);
  const relatedSchemas = [
    ...extractRelatedSchemasFromOperation(domain, found.details),
    ...extractResponseSchemaCardFromOperation(domain, found.details)
  ].filter((item, idx, arr) => arr.findIndex((x) => x.title === item.title) === idx);

  return res.json({
    domain,
    method: found.method,
    path: found.path,
    summary: found.details.summary || '',
    description: found.details.description || '',
    operationId: found.details.operationId || '',
    tags: Array.isArray(found.details.tags) ? found.details.tags : [],
    parameters,
    related_schema_cards: relatedSchemas,
    request_example: extractRequestExampleFromOperation(domain, found.details),
    response_example: extractResponseExampleFromOperation(domain, found.details),
    raw_openapi: found.details
  });
});

app.get('/api/openapi-schema', (req, res) => {
  const domain = String(req.query.domain || '').trim();
  const name = String(req.query.name || '');

  if (!domain || !name) {
    return res.status(400).json({ error: 'domain and name are required.' });
  }

  const found = findOpenApiSchemaByName(domain, name);
  if (!found) {
    return res.status(404).json({ error: `Schema not found for domain: ${domain}` });
  }

  const related = Array.from(collectSchemaNamesFromRefs(found.raw, new Set())).map((schemaName) => ({
    title: schemaName,
    description: getOpenApiSchemaMap(domain)[schemaName]?.description || 'Related schema',
    prompt: `Show schema ${schemaName}`
  }));

  return res.json({
    domain,
    name: found.name,
    raw_schema: found.raw,
    related_schema_cards: related
  });
});

/* --------------------------------------------------
 * MAIN APP ROUTES
 * -------------------------------------------------- */

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'star-ai-intelligence-portal',
    model: MODEL,
    has_openai_key: Boolean(OPENAI_API_KEY),
    domains: Object.fromEntries(
      Object.keys(DOMAIN_SPEC_PATHS).map((domain) => [
        domain,
        {
          openapi_loaded: Boolean(getOpenApiSpec(domain)),
          openapi_source: getOpenApiSourcePath(domain)
        }
      ])
    )
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
        tool_arguments: { domain_name: domain, schema_name: requestedSchemaName, audience }
      });
    }

    if (domain && wantsSubApiDesign(message)) {
      const result = await buildSubApiResponse(domain, message, audience);
      return res.json({
        ...result,
        tool_name: 'sub_api_pipeline',
        tool_arguments: { domain_name: domain, audience }
      });
    }

    if (domain && requestedEndpoint) {
      const result = await buildEndpointDetailResponse(domain, requestedEndpoint, audience);
      return res.json({
        ...result,
        tool_name: 'endpoint_detail_pipeline',
        tool_arguments: { domain_name: domain, endpoint: requestedEndpoint, audience }
      });
    }

    if (domain && wantsCapabilities(message)) {
      const result = await buildCapabilitiesResponse(domain, audience);
      return res.json({
        ...result,
        tool_name: 'business_capability_navigator',
        tool_arguments: { domain_name: domain, audience }
      });
    }

    if (domain && wantsSchemas(message)) {
      const result = await buildSchemaListResponse(domain, audience);
      return res.json({
        ...result,
        tool_name: 'schema_pipeline',
        tool_arguments: { domain_name: domain, audience }
      });
    }

    if (domain && wantsEndpoints(message)) {
      const result = await buildEndpointListResponse(domain, audience);
      return res.json({
        ...result,
        tool_name: 'endpoint_pipeline',
        tool_arguments: { domain_name: domain, audience }
      });
    }

    if (domain && wantsOverview(message)) {
      const result = await buildOverviewResponse(domain, message, audience);
      return res.json({
        ...result,
        tool_name: 'overview_pipeline',
        tool_arguments: { domain_name: domain, audience }
      });
    }

    if (client) {
      const ai = await withTimeout(
        client.responses.create({
          model: MODEL,
          input: `Audience: ${audience}
User request: ${message}

Answer clearly and concisely in the context of STAR automotive APIs.`
        }),
        12000,
        'OpenAI answer generation'
      );

      return res.json({
        answer: ai.output_text,
        sections: extractSections(ai.output_text),
        audience,
        capability_cards: [],
        endpoint_cards: [],
        schema_cards: [],
        workflow_map: [],
        tool_name: 'general_answer',
        tool_arguments: { audience },
        source: 'openai'
      });
    }

    return res.json({
      answer: 'OpenAI is not configured, and the request did not match a known domain-specific path.',
      sections: extractSections('1. Direct answer\nOpenAI is not configured, and the request did not match a known domain-specific path.'),
      audience,
      capability_cards: [],
      endpoint_cards: [],
      schema_cards: [],
      workflow_map: [],
      tool_name: 'general_answer',
      tool_arguments: { audience },
      source: 'fallback'
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
  console.log(`STAR multi-spec server running on ${PORT}`);
});
