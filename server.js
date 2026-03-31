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

const ROOT = process.cwd();

const DOMAIN_SPEC_PATHS = {
  appointment:
    process.env.APPOINTMENT_OPENAPI_SPEC_PATH ||
    path.join(ROOT, 'openapi', 'appointment-api.yaml'),
  'multi-point-inspection':
    process.env.MPI_OPENAPI_SPEC_PATH ||
    path.join(ROOT, 'openapi', 'multi-point-inspection-api.yaml')
};

function loadYamlFile(filePath) {
  try {
    const resolved = path.resolve(filePath);

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
    prompt: `Show schema ${name} in the ${domain} API`
  }));
}

function extractResponseSchemaCardFromOperation(domain, op) {
  const responses = op?.responses || {};
  const preferred =
    responses['200'] ||
    responses['201'] ||
    responses['202'] ||
    Object.values(responses)[0];

  if (!preferred) return [];

  const content = findPrimaryJsonContent(preferred.content);
  if (!content?.schema) return [];

  const found = collectSchemaNamesFromRefs(content.schema, new Set());
  return Array.from(found).map((name) => ({
    title: name,
    description: getOpenApiSchemaMap(domain)[name]?.description || 'Response schema',
    prompt: `Show schema ${name} in the ${domain} API`
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
  const preferred =
    responses['200'] ||
    responses['201'] ||
    responses['202'] ||
    Object.values(responses)[0];

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
 * BUILT-IN CONTENT
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

function getBuiltInCapabilityCards(domain, audience = 'business') {
  if (domain === 'multi-point-inspection') {
    if (audience === 'architecture') {
      return [
        {
          title: 'Inspection aggregate boundary',
          description: 'Shows how inspection acts as the central domain object connecting findings, recommendations, and approval outcomes.',
          prompt: 'Explain the inspection aggregate boundary in the multi-point-inspection API'
        },
        {
          title: 'Cross-domain handoff',
          description: 'Highlights how inspection context connects to intake, appointment, and downstream service execution.',
          prompt: 'Explain cross-domain handoffs in the multi-point-inspection API'
        },
        {
          title: 'Lifecycle state model',
          description: 'Describes the major inspection lifecycle transitions and where they matter architecturally.',
          prompt: 'Explain the lifecycle state model in the multi-point-inspection API'
        }
      ];
    }

    if (audience === 'technical') {
      return [
        {
          title: 'Start inspection workflow',
          description: 'Identify how inspections are created, initialized, and tied to upstream context.',
          prompt: 'Explain how to start the inspection workflow in the multi-point-inspection API'
        },
        {
          title: 'Review findings and media',
          description: 'Inspect how findings, notes, and evidence objects are modeled and retrieved.',
          prompt: 'Explain how to review findings and media in the multi-point-inspection API'
        },
        {
          title: 'Capture customer approval',
          description: 'Understand how approval or decline decisions are represented and linked to recommendations.',
          prompt: 'Explain how customer approval works in the multi-point-inspection API'
        }
      ];
    }

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
    if (audience === 'architecture') {
      return [
        {
          title: 'Scheduling boundary',
          description: 'Defines the Appointment domain as the scheduling and intake boundary before downstream service workflows.',
          prompt: 'Explain the scheduling boundary in the Appointment API'
        },
        {
          title: 'Reference handoff model',
          description: 'Shows how appointment data can pass vehicle, customer, and requested-service context to downstream domains.',
          prompt: 'Explain reference handoffs in the Appointment API'
        },
        {
          title: 'Lifecycle and coordination',
          description: 'Highlights lifecycle state and coordination points that matter architecturally.',
          prompt: 'Explain the lifecycle and coordination model in the Appointment API'
        }
      ];
    }

    if (audience === 'technical') {
      return [
        {
          title: 'Schedule appointment',
          description: 'Create and retrieve service appointments using the core appointment resource model.',
          prompt: 'Explain how to schedule and manage appointments in the Appointment API'
        },
        {
          title: 'Confirm service needs',
          description: 'Inspect how requested service and intake context are modeled.',
          prompt: 'Explain how a service advisor confirms service needs using the Appointment API'
        },
        {
          title: 'Retrieve appointment state',
          description: 'Understand the read path for a single appointment and its status.',
          prompt: 'Explain how to retrieve appointment state in the Appointment API'
        }
      ];
    }

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
      },
      {
        title: 'Prepare service lane',
        description: 'Use appointment context to support dealership operational readiness.',
        prompt: 'Explain how the Appointment API supports service-lane preparation'
      }
    ];
  }

  return [];
}

function getBuiltInWorkflowMap(domain, audience = 'business') {
  if (domain === 'multi-point-inspection') {
    if (audience === 'architecture') {
      return [
        { step: 'Intake linkage', detail: 'Inspection context originates from upstream intake, appointment, or repair context', prompt: 'Explain intake linkage in the multi-point-inspection API' },
        { step: 'Inspection aggregate', detail: 'Inspection serves as the anchor entity for findings and recommendation flow', prompt: 'Explain the inspection aggregate in the multi-point-inspection API' },
        { step: 'Decision boundary', detail: 'Customer approval becomes the handoff point into downstream execution', prompt: 'Explain the decision boundary in the multi-point-inspection API' }
      ];
    }

    return [
      { step: 'Start inspection', detail: 'Launch MPI workflow tied to intake or RO context', prompt: 'Explain how to start the inspection workflow in the multi-point-inspection API' },
      { step: 'Capture findings', detail: 'Technician records results, notes, and media', prompt: 'Explain how findings are captured in the multi-point-inspection API' },
      { step: 'Build recommendations', detail: 'Advisor prepares customer-facing repair guidance', prompt: 'Explain how recommendations are built in the multi-point-inspection API' },
      { step: 'Get approval', detail: 'Customer decisions are captured and tracked', prompt: 'Explain customer approval flow in the multi-point-inspection API' }
    ];
  }

  if (domain === 'appointment') {
    if (audience === 'architecture') {
      return [
        { step: 'Schedule boundary', detail: 'Appointment defines the upstream scheduling and intake domain', prompt: 'Explain the scheduling boundary in the Appointment API' },
        { step: 'Reference handoff', detail: 'Appointment passes context to downstream service operations', prompt: 'Explain reference handoffs in the Appointment API' },
        { step: 'Operational coordination', detail: 'Appointment state supports dealer coordination and readiness', prompt: 'Explain operational coordination in the Appointment API' }
      ];
    }

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
It gives consumers a collection-level entry point into the MPI domain.`,
      'GET /INSPECTIONS/{ID}': `1. Direct answer
GET /inspections/{id} retrieves a specific inspection.

2. Key endpoint details
This endpoint is used to inspect the full state of a single inspection, including status, relationships, and downstream context.

3. Why it matters
It is the most direct path for reading a specific MPI entity.`,
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
 * DETERMINISTIC CAPABILITY DETAIL FIX
 * -------------------------------------------------- */

function getBuiltInCapabilityDetail(domain, prompt, audience = 'business') {
  const p = String(prompt || '').toLowerCase();

  if (domain === 'multi-point-inspection') {
    if (p.includes('start inspection')) {
      if (audience === 'architecture') {
        return `1. Direct answer
The inspection workflow begins when an inspection context is created and linked to an upstream service event such as intake, appointment, or repair order.

2. Key details
Architecturally, this is the point where the inspection aggregate is established and becomes the anchor for findings, recommendation flow, and downstream approval state.

3. Useful next steps
Review the Inspection schema and the create or start inspection endpoints to understand how the workflow is initialized.`;
      }

      if (audience === 'technical') {
        return `1. Direct answer
The inspection workflow starts when the system creates or initializes an Inspection resource for a vehicle service event.

2. Key details
Technical consumers should inspect how the inspection identity is created, what upstream references are required, and how initial workflow status is represented.

3. Useful next steps
Review the Inspection schema and the POST /inspections endpoint to understand initialization inputs and response structure.`;
      }

      return `1. Direct answer
The inspection workflow starts when a dealership begins a multi-point inspection for a vehicle visit.

2. Key details
This typically ties the inspection to a customer visit, vehicle, and service context so technicians and advisors can track findings and next steps.

3. Useful next steps
Explore the Inspection schema and related workflow endpoints to see how the process is launched and managed.`;
    }

    if (p.includes('track inspection progress')) {
      return `1. Direct answer
Inspection progress is tracked through inspection status and workflow state.

2. Key details
This helps teams see where the inspection stands, what has been completed, and when advisor follow-up can begin.

3. Useful next steps
Review inspection retrieval endpoints and status-related schemas.`;
    }

    if (p.includes('review findings and media')) {
      return `1. Direct answer
Findings and media are reviewed after technician observations are captured during the inspection.

2. Key details
This stage turns raw condition evidence into advisor-usable information for explanation and recommendation.

3. Useful next steps
Inspect the Finding and InspectionMedia schemas and related retrieval endpoints.`;
    }

    if (p.includes('customer approval') || p.includes('capture customer approval')) {
      return `1. Direct answer
Customer approval captures whether recommended work is accepted or declined.

2. Key details
This creates the business decision point between inspection findings and downstream execution.

3. Useful next steps
Inspect approval-related schemas and endpoints tied to recommendation outcomes.`;
    }

    if (p.includes('close and publish results')) {
      return `1. Direct answer
Closing and publishing results finalizes the inspection and makes outcomes available for follow-through.

2. Key details
This marks the end of the inspection workflow and supports communication, reporting, and downstream service actions.

3. Useful next steps
Review final inspection state and output-oriented schemas or endpoints.`;
    }

    if (p.includes('inspection aggregate')) {
      return `1. Direct answer
The Inspection object acts as the aggregate root of the MPI domain.

2. Key details
It anchors findings, recommendation flow, and approval-related outcomes as the inspection progresses.

3. Useful next steps
Review the Inspection schema and its related entities to see how the model is structured.`;
    }

    if (p.includes('cross-domain handoff') || p.includes('handoff')) {
      return `1. Direct answer
The MPI domain receives upstream service context and produces downstream decision outcomes.

2. Key details
It typically consumes intake, appointment, or repair context and hands approved work forward into later service execution processes.

3. Useful next steps
Review domain boundaries, related references, and workflow transition points.`;
    }

    if (p.includes('lifecycle state model')) {
      return `1. Direct answer
The lifecycle state model describes how an inspection moves from initialization to completion.

2. Key details
It provides structure for progress tracking, findings capture, recommendation review, approval, and finalization.

3. Useful next steps
Inspect status-related schemas and endpoints that expose inspection progression.`;
    }

    if (p.includes('decision boundary')) {
      return `1. Direct answer
The decision boundary occurs when customer approval determines which inspection outcomes move forward.

2. Key details
This is the point where recommendation review transitions into downstream execution decisions.

3. Useful next steps
Review approval and recommendation-related structures in the API.`;
    }

    if (p.includes('intake linkage')) {
      return `1. Direct answer
Intake linkage refers to the upstream service context connected to an inspection.

2. Key details
Architecturally, this allows the inspection domain to participate in a larger dealership service flow without owning all upstream scheduling or intake responsibilities.

3. Useful next steps
Review upstream references associated with the Inspection resource.`;
    }
  }

  if (domain === 'appointment') {
    if (p.includes('schedule appointment') || p.includes('manage appointments')) {
      return `1. Direct answer
The appointment workflow starts when a service appointment is created for a customer and vehicle.

2. Key details
This establishes scheduling, requested service context, and readiness for downstream dealership operations.

3. Useful next steps
Inspect the Appointment schema and main appointment endpoints.`;
    }

    if (p.includes('confirm service needs')) {
      return `1. Direct answer
Confirming service needs means validating the requested work, timing, and intake context of an appointment.

2. Key details
This helps advisors ensure the service request is ready for operational planning and downstream handoff.

3. Useful next steps
Review requested-service related schemas and appointment retrieval endpoints.`;
    }

    if (p.includes('prepare service lane')) {
      return `1. Direct answer
Preparing the service lane means using appointment information to support operational readiness.

2. Key details
This includes aligning timing, requested services, and intake context before the vehicle arrives or is processed.

3. Useful next steps
Review appointment status and requested-service related structures.`;
    }

    if (p.includes('scheduling boundary')) {
      return `1. Direct answer
The Appointment API defines the upstream scheduling and intake boundary.

2. Key details
Architecturally, it owns appointment creation and scheduling context before downstream service workflows begin.

3. Useful next steps
Review appointment lifecycle and related references to downstream domains.`;
    }

    if (p.includes('reference handoff')) {
      return `1. Direct answer
Reference handoff describes how appointment data passes customer, vehicle, and service context to downstream domains.

2. Key details
This supports continuity between scheduling and later dealership service operations.

3. Useful next steps
Inspect the Appointment schema and its key references.`;
    }

    if (p.includes('lifecycle and coordination') || p.includes('operational coordination')) {
      return `1. Direct answer
Lifecycle and coordination in the Appointment API describe how scheduling state supports dealer operations.

2. Key details
This includes creation, confirmation, readiness, and coordination touchpoints that prepare downstream work.

3. Useful next steps
Review appointment status and timing-related schema structures.`;
    }

    if (p.includes('retrieve appointment state')) {
      return `1. Direct answer
Retrieving appointment state means reading the current details and status of a single appointment.

2. Key details
This is important for operational lookup, status awareness, and advisor workflow support.

3. Useful next steps
Review GET /appointments/{id} and the AppointmentStatus schema.`;
    }
  }

  return null;
}

/* --------------------------------------------------
 * SAFE TEXT HELPERS
 * -------------------------------------------------- */

async function maybeImproveText(prompt, fallbackText, timeoutMs = 12000) {
  if (!client) return fallbackText;

  try {
    const response = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: prompt
      }),
      timeoutMs,
      'OpenAI answer generation'
    );
    return response.output_text || fallbackText;
  } catch {
    return fallbackText;
  }
}

/* --------------------------------------------------
 * RESPONSE BUILDERS
 * -------------------------------------------------- */

async function buildOverviewResponse(domain, message, audience) {
  const fallbackText = getAudienceOverview(domain, audience);

  const answer = await maybeImproveText(
    `Audience: ${audience}
Domain: ${domain}
User request: ${message}

Base overview:
${fallbackText}

Write a concise structured response with:
1. Direct answer
2. Core responsibilities
3. Useful next steps

Keep it under 120 words.`,
    fallbackText,
    16000
  );

  const ops = getOpenApiOperationsForDomain(domain);
  const schemas = getOpenApiSchemasForDomain(domain);

  return {
    answer,
    sections: extractSections(answer),
    audience,
    capability_cards: getBuiltInCapabilityCards(domain, audience),
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
    workflow_map: getBuiltInWorkflowMap(domain, audience),
    progressive: false,
    source: ops.length || schemas.length ? 'openapi' : 'fallback'
  };
}

async function buildCapabilitiesResponse(domain, audience) {
  const fallbackText =
    audience === 'technical'
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

  const capabilitySummary = getBuiltInCapabilityCards(domain, audience)
    .map((c) => `- ${c.title}: ${c.description}`)
    .join('\n');

  const answer = await maybeImproveText(
    `Audience: ${audience}
Domain: ${domain}

Known capability cards:
${capabilitySummary}

Base response:
${fallbackText}

Write a short structured response with:
1. Direct answer
2. Key details
3. Useful next steps

Keep it under 100 words.`,
    fallbackText,
    16000
  );

  const ops = getOpenApiOperationsForDomain(domain);
  const schemas = getOpenApiSchemasForDomain(domain);

  return {
    answer,
    sections: extractSections(answer),
    audience,
    capability_cards: getBuiltInCapabilityCards(domain, audience),
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
    workflow_map: getBuiltInWorkflowMap(domain, audience),
    progressive: false,
    source: ops.length || schemas.length ? 'openapi' : 'fallback'
  };
}

async function buildSubApiResponse(domain, message, audience) {
  const ops = getOpenApiOperationsForDomain(domain);
  const getOps = ops.filter((op) => op.method === 'GET');

  const fallbackText = `1. Direct answer
A focused sub-API can be derived for the ${domain} domain.

2. Proposed resources or endpoints
Start with a reduced set of read-oriented GET operations that expose the main collection and entity retrieval flows.

3. Proposed schemas
Reuse the core read models and supporting reference schemas needed for consumers.

4. Design notes
Prefer a narrow read-only surface, stable identifiers, and minimal cross-domain coupling.

5. Suggested next steps
Select the GET endpoints you want to keep and then trim the schema surface to only what those responses need.`;

  const answer = await maybeImproveText(
    `Audience: ${audience}
Domain: ${domain}
User request: ${message}

Available GET operations:
${JSON.stringify(
  getOps.map((op) => ({
    method: op.method,
    path: op.path,
    summary: op.summary,
    description: op.description
  })),
  null,
  2
)}

Base response:
${fallbackText}

Write a structured sub-API proposal in 5 short sections.`,
    fallbackText,
    18000
  );

  return {
    answer,
    sections: extractSections(answer),
    audience,
    capability_cards: getBuiltInCapabilityCards(domain, audience),
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
    workflow_map: getBuiltInWorkflowMap(domain, audience),
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
    capability_cards: getBuiltInCapabilityCards(domain, audience),
    endpoint_cards: [],
    schema_cards: schemas.length
      ? schemas.map((s) => ({
          title: s.name,
          description: s.description || 'View schema details',
          prompt: `Show schema ${s.name} in the ${domain} API`
        }))
      : getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain, audience),
    progressive: !schemas.length,
    source: schemas.length ? 'openapi' : 'fallback'
  };
}

async function buildSchemaDetailResponse(domain, schemaName, audience) {
  const found = findOpenApiSchemaByName(domain, schemaName);

  if (found) {
    const fallbackText = `1. Direct answer
${found.name} is a schema in the ${domain} API.

2. Key schema details
This schema is available from the OpenAPI definition and can be inspected in raw form.

3. Why it matters
It represents part of the structural contract of the domain.

4. Useful next steps
Review related schemas and endpoints that reference it.`;

    const answer = await maybeImproveText(
      `Audience: ${audience}
Domain: ${domain}
Schema name: ${found.name}
Schema data:
${JSON.stringify(found.raw, null, 2)}

Base response:
${fallbackText}

Write:
1. Direct answer
2. Key schema details
3. Why it matters
4. Useful next steps

Keep it concise.`,
      fallbackText,
      16000
    );

    const related = Array.from(collectSchemaNamesFromRefs(found.raw, new Set())).map((name) => ({
      title: name,
      description: getOpenApiSchemaMap(domain)[name]?.description || 'Related schema',
      prompt: `Show schema ${name} in the ${domain} API`
    }));

    return {
      answer,
      sections: extractSections(answer),
      audience,
      schema_title: found.name,
      raw_schema: found.raw,
      raw_schema_format: 'json',
      related_schema_cards: related,
      capability_cards: getBuiltInCapabilityCards(domain, audience),
      endpoint_cards: getBuiltInEndpointCards(domain),
      schema_cards: getBuiltInSchemaCards(domain),
      workflow_map: getBuiltInWorkflowMap(domain, audience),
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
    capability_cards: getBuiltInCapabilityCards(domain, audience),
    endpoint_cards: getBuiltInEndpointCards(domain),
    schema_cards: getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain, audience),
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
    capability_cards: getBuiltInCapabilityCards(domain, audience),
    endpoint_cards: ops.length
      ? ops.map((op) => ({
          title: `${op.method} ${op.path}`,
          description: op.summary || op.description || 'API endpoint',
          prompt: `Explain ${op.method} ${op.path} for the ${domain} API`
        }))
      : getBuiltInEndpointCards(domain),
    schema_cards: getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain, audience),
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

    const fallbackText = `1. Direct answer
${openApiEndpoint.method} ${openApiEndpoint.path} is a defined endpoint in the ${domain} API.

2. Key endpoint details
${op.summary || op.description || 'This endpoint is part of the domain API surface.'}

3. Why it matters
This endpoint is relevant to consumers who need direct access to this resource or workflow step.

4. Useful next steps
Review the parameters, related schemas, examples, and raw OpenAPI definition.`;

    const answer = await maybeImproveText(
      `Audience: ${audience}
Domain: ${domain}
Endpoint: ${openApiEndpoint.method} ${openApiEndpoint.path}

Summary: ${op.summary || ''}
Description: ${op.description || ''}
Parameters:
${JSON.stringify(parameters, null, 2)}

Base response:
${fallbackText}

Write:
1. Direct answer
2. Key endpoint details
3. Why it matters
4. Useful next steps

Keep it concise.`,
      fallbackText,
      16000
    );

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
      capability_cards: getBuiltInCapabilityCards(domain, audience),
      endpoint_cards: getBuiltInEndpointCards(domain),
      schema_cards: getBuiltInSchemaCards(domain),
      workflow_map: getBuiltInWorkflowMap(domain, audience),
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

  const answer =
    builtIn ||
    `1. Direct answer
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
    capability_cards: getBuiltInCapabilityCards(domain, audience),
    endpoint_cards: getBuiltInEndpointCards(domain),
    schema_cards: getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain, audience),
    progressive: true,
    source: 'fallback'
  };
}

/* --------------------------------------------------
 * OPENAPI EXPLORER ROUTES
 * -------------------------------------------------- */

app.get('/api/version', (_req, res) => {
  res.json({
    version: 'multi-spec-registry-demo-safe',
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

    if (domain) {
      const builtInCapabilityDetail = getBuiltInCapabilityDetail(domain, message, audience);
      if (builtInCapabilityDetail) {
        const ops = getOpenApiOperationsForDomain(domain);
        const schemas = getOpenApiSchemasForDomain(domain);

        return res.json({
          answer: builtInCapabilityDetail,
          sections: extractSections(builtInCapabilityDetail),
          audience,
          capability_cards: getBuiltInCapabilityCards(domain, audience),
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
          workflow_map: getBuiltInWorkflowMap(domain, audience),
          progressive: false,
          tool_name: 'capability_detail_pipeline',
          tool_arguments: { domain_name: domain, audience },
          source: ops.length || schemas.length ? 'openapi' : 'fallback'
        });
      }
    }

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
  console.log(`STAR multi-spec demo-safe server running on ${PORT}`);
});
