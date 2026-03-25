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
const MODEL = process.env.OPENAI_MODEL || 'gpt-5';

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
        'Accept': 'application/json, text/event-stream'
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
  const text = await withTimeout(
    response.text(),
    10000,
    `MCP ${method} body read`
  );

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

function normalizeLooseText(message = '') {
  return String(message)
    .toLowerCase()
    .replace(/inspectios/g, 'inspections')
    .replace(/opertion/g, 'operation')
    .replace(/operationonly/g, 'operation only')
    .replace(/subapi/g, 'sub api')
    .replace(/\s+/g, ' ')
    .trim();
}

function detectDomainFromEndpoint(message = '') {
  const m = normalizeLooseText(message);

  if (
    m.includes('/inspections') ||
    m.includes('/inspection') ||
    m.includes('/inspections/{id}') ||
    m.includes('/inspections/')
  ) {
    return 'multi-point-inspection';
  }

  if (
    m.includes('/appointments') ||
    m.includes('/appointment') ||
    m.includes('/appointments/{id}') ||
    m.includes('/appointments/')
  ) {
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
    m.includes('path')
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
    if (match && match[1]) {
      return match[1].trim();
    }
  }

  return null;
}

function normalizeEndpointText(endpoint = '') {
  return String(endpoint)
    .toUpperCase()
    .replace('/INSPECTIOS', '/INSPECTIONS')
    .replace('/INSPECTIO', '/INSPECTION')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractRequestedEndpoint(message = '') {
  const text = String(message || '').trim();
  const match = text.match(/(GET|POST|PUT|PATCH|DELETE)\s+\/[A-Za-z0-9/_{}-]+/i);

  if (!match) return null;

  return normalizeEndpointText(match[0]);
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

function normalizeEndpointTitle(endpointTitle = '') {
  return normalizeEndpointText(endpointTitle);
}

/* --------------------------------------------------
 * BUILT-IN DOMAIN INTELLIGENCE
 * -------------------------------------------------- */

const DOMAIN_CONFIG = {
  appointment: {
    business_overview: `1. What it is
The Appointment API represents the dealership service scheduling and intake layer. It helps coordinate customer bookings, requested services, timing expectations, and operational readiness before the vehicle enters the service lane.

2. Core concepts
The API centers on the service appointment as the planning object that connects customer intent, vehicle context, dealership availability, and requested work.

3. Main resources
Typical resources include the appointment itself, requested services, customer and vehicle references, timing details, and status information needed by dealership systems.

4. Typical workflow
A customer books service, the dealership validates the request, the advisor confirms timing and service intent, and the service lane uses the appointment context to prepare intake and execution.

5. What to explore next
Look at business capabilities, representative endpoints, and schemas related to scheduling, intake, and service preparation.`,

    technical_overview: `1. What it is
The Appointment API is the scheduling and intake domain for dealership service operations. It models appointment creation, requested service context, party and vehicle references, timing, and workflow state.

2. Technical model
A typical implementation treats Appointment as a bounded-context resource with references to customer, vehicle, requested services, and scheduling metadata. The API helps move from customer intent into executable service-lane context.

3. Resource families
Typical technical resources include appointment entities, request details, time windows, references, and status fields needed for intake orchestration.

4. Integration concerns
Technical consumers care about identifiers, party and vehicle references, status transitions, and how appointment context flows into downstream service systems.

5. What to explore next
Inspect schema cards, endpoint cards, required fields, and object relationships.`,

    architecture_overview: `1. Bounded context role
The Appointment API sits at the front of the dealership service journey. It owns service-booking and pre-intake coordination concerns.

2. Boundary
Its responsibility is not repair execution or detailed inspection results. Its role is to create the operational context that downstream domains consume.

3. Cross-domain relationships
Appointment typically connects to customer, vehicle, scheduling, intake, and downstream service execution contexts.

4. Architectural significance
This domain is valuable because it separates scheduling intent from execution concerns while still providing the references needed for downstream orchestration.

5. What to explore next
Review bounded-context relationships, key references, and how Appointment feeds inspection and repair workflows.`,

    capability_cards: [
      {
        title: 'Schedule appointment',
        description: 'Create and manage service appointments for customers and vehicles.',
        business_guidance: 'This capability covers the initial booking process and represents the front door into service operations.',
        advisor_actions: [
          'Review requested services and timing',
          'Confirm customer and vehicle context',
          'Make sure appointment data supports intake readiness'
        ],
        customer_impact: 'Customers get a clearer booking experience and more accurate service expectations.',
        workflow_stage: 'Scheduling',
        technical_notes: 'Developers should focus on appointment identifiers, requested service structures, time windows, and party or vehicle references.',
        data_model_notes: 'The core model typically includes appointment identity, service intent, scheduling metadata, and supporting references.',
        integration_notes: 'This capability commonly feeds service-lane preparation, intake workflows, and downstream service domains.',
        prompt: 'Explain how to schedule and manage appointments in the Appointment API'
      },
      {
        title: 'Confirm service needs',
        description: 'Validate requested services, timing, and customer expectations.',
        business_guidance: 'This capability helps the dealership turn a raw booking into an operationally useful service request.',
        advisor_actions: [
          'Validate requested work',
          'Check timing and operational fit',
          'Reduce ambiguity before service intake'
        ],
        customer_impact: 'Customers receive more accurate service planning and fewer surprises at intake.',
        workflow_stage: 'Pre-intake validation',
        technical_notes: 'Technical consumers should examine how requested services, notes, and appointment state are represented and updated.',
        data_model_notes: 'This area often involves validation rules, optional versus required fields, and how pre-intake details are normalized.',
        integration_notes: 'This step often influences downstream systems that consume intake-ready appointment context.',
        prompt: 'Explain how a service advisor confirms service needs using the Appointment API'
      },
      {
        title: 'Prepare service lane',
        description: 'Use appointment context to prepare the dealership service workflow.',
        business_guidance: 'This capability connects the appointment to the dealership operating model and helps the service lane prepare before the customer arrives.',
        advisor_actions: [
          'Review upcoming appointments',
          'Use appointment data to prepare intake',
          'Coordinate service-lane readiness'
        ],
        customer_impact: 'Customers experience smoother intake and better operational coordination.',
        workflow_stage: 'Service preparation',
        technical_notes: 'Technical teams should look at how appointment data is exposed for operational consumers and how readiness-related status is modeled.',
        data_model_notes: 'Key elements usually include schedule context, requested services, and references needed by intake systems.',
        integration_notes: 'This capability commonly interacts with scheduling, check-in, and service execution contexts.',
        prompt: 'Explain how the Appointment API helps prepare the service lane'
      }
    ],

    workflow_map: [
      { step: 'Schedule appointment', detail: 'Customer booking and advisor intake', prompt: 'Explain the scheduling flow of the Appointment API' },
      { step: 'Confirm service needs', detail: 'Advisor validates request and timing', prompt: 'Explain advisor validation workflow in the Appointment API' },
      { step: 'Prepare service lane', detail: 'Appointment data supports dealer operations', prompt: 'Explain how the Appointment API supports service-lane preparation' }
    ],

    schema_cards: [
      { title: 'Appointment', description: 'Core appointment entity for dealership service scheduling.', prompt: 'Show schema Appointment in the Appointment API' },
      { title: 'RequestedService', description: 'Represents requested service work tied to an appointment.', prompt: 'Show schema RequestedService in the Appointment API' },
      { title: 'AppointmentStatus', description: 'Represents appointment lifecycle or state.', prompt: 'Show schema AppointmentStatus in the Appointment API' },
      { title: 'VehicleReference', description: 'Vehicle identity and linkage used by the appointment domain.', prompt: 'Show schema VehicleReference in the Appointment API' },
      { title: 'PartyReference', description: 'Customer or related party linkage used in the appointment domain.', prompt: 'Show schema PartyReference in the Appointment API' },
      { title: 'TimeSlot', description: 'Scheduling or timing structure for appointment planning.', prompt: 'Show schema TimeSlot in the Appointment API' },
      { title: 'AppointmentNote', description: 'Represents notes or comments attached to the appointment.', prompt: 'Show schema AppointmentNote in the Appointment API' },
      { title: 'ServiceAdvisorReference', description: 'Represents advisor assignment or linkage.', prompt: 'Show schema ServiceAdvisorReference in the Appointment API' },
      { title: 'DealerReference', description: 'Represents dealer identity or organizational reference.', prompt: 'Show schema DealerReference in the Appointment API' },
      { title: 'AppointmentChannel', description: 'Represents appointment booking channel or source.', prompt: 'Show schema AppointmentChannel in the Appointment API' }
    ],

    endpoint_cards: [
      { title: 'GET /appointments', description: 'List or search appointments.', prompt: 'Explain GET /appointments for the Appointment API' },
      { title: 'POST /appointments', description: 'Create a new appointment.', prompt: 'Explain POST /appointments for the Appointment API' },
      { title: 'GET /appointments/{id}', description: 'Retrieve a specific appointment.', prompt: 'Explain GET /appointments/{id} for the Appointment API' },
      { title: 'PATCH /appointments/{id}', description: 'Update an existing appointment.', prompt: 'Explain PATCH /appointments/{id} for the Appointment API' },
      { title: 'GET /appointments/{id}/requested-services', description: 'View requested services tied to an appointment.', prompt: 'Explain GET /appointments/{id}/requested-services for the Appointment API' }
    ],

    schema_details: {
      Appointment: {
        title: 'Appointment',
        body: `1. Direct answer
Appointment is the core entity of the Appointment API and represents a dealership service booking or planned visit.

2. Key schema details
This schema usually contains customer and vehicle references, service intent, schedule context, and lifecycle state.

3. Why it matters
It is the anchor object for the scheduling and intake domain.`
      },
      RequestedService: {
        title: 'RequestedService',
        body: `1. Direct answer
RequestedService represents service work requested as part of an appointment.

2. Key schema details
It typically carries structured service intent and may be linked to appointment context, customer expectations, and intake planning.

3. Why it matters
It helps translate customer booking intent into operational service context.`
      },
      AppointmentStatus: {
        title: 'AppointmentStatus',
        body: `1. Direct answer
AppointmentStatus represents lifecycle or workflow state for an appointment.

2. Key schema details
This can be modeled as a status field, enum-like object, or supporting schema depending on implementation.

3. Why it matters
Status is important for scheduling flow, intake readiness, and operational coordination.`
      }
    },

    raw_schemas: {
      Appointment: {
        type: 'object',
        required: ['appointment_id'],
        properties: {
          appointment_id: { type: 'string' },
          appointment_status: { $ref: '#/components/schemas/AppointmentStatus' },
          vehicle_reference: { $ref: '#/components/schemas/VehicleReference' },
          party_reference: { $ref: '#/components/schemas/PartyReference' },
          time_slot: { $ref: '#/components/schemas/TimeSlot' },
          requested_services: {
            type: 'array',
            items: { $ref: '#/components/schemas/RequestedService' }
          }
        }
      },
      RequestedService: {
        type: 'object',
        properties: {
          requested_service_id: { type: 'string' },
          service_name: { type: 'string' },
          service_description: { type: 'string' }
        }
      },
      AppointmentStatus: {
        type: 'string',
        enum: ['scheduled', 'confirmed', 'in_progress', 'completed', 'cancelled']
      }
    },

    endpoint_details: {
      'GET /APPOINTMENTS': `1. Direct answer
GET /appointments lists or searches appointments.

2. Key endpoint details
This endpoint supports schedule views, search flows, and operational appointment lookup.

3. Why it matters
It is the main collection endpoint for the Appointment API.`,
      'POST /APPOINTMENTS': `1. Direct answer
POST /appointments creates a new appointment.

2. Key endpoint details
This endpoint captures booking intent, customer and vehicle context, and requested service details.

3. Why it matters
It is the main entry point for appointment creation.`,
      'GET /APPOINTMENTS/{ID}': `1. Direct answer
GET /appointments/{id} retrieves a specific appointment.

2. Key endpoint details
This endpoint returns the state and details of one appointment instance.

3. Why it matters
It is the primary read path for single-appointment lookup.`,
      'PATCH /APPOINTMENTS/{ID}': `1. Direct answer
PATCH /appointments/{id} updates an existing appointment.

2. Key endpoint details
This endpoint is typically used for status updates, timing adjustments, or request refinement.

3. Why it matters
It supports lifecycle management of appointment entities.`,
      'GET /APPOINTMENTS/{ID}/REQUESTED-SERVICES': `1. Direct answer
GET /appointments/{id}/requested-services retrieves requested services tied to an appointment.

2. Key endpoint details
This endpoint exposes the service intent associated with a booking.

3. Why it matters
It helps technical users understand how service intent is represented separately from the appointment root.`
    }
  },

  'multi-point-inspection': {
    business_overview: `1. What it is
The Multi-Point Inspection API is a dealership service workflow domain that supports structured vehicle health checks. It covers the lifecycle from launching an inspection through technician findings, recommendations, customer decisions, and final completion.

2. Core concepts
The domain revolves around the inspection itself, what is being inspected, what was found, and what action should be recommended or approved.

3. Main resources
Typical resources include the inspection record, findings, condition evidence, media, recommendations, approval outcomes, and workflow status.

4. Typical workflow
An inspection is started from service intake or repair-order context. The technician records findings and supporting evidence. The advisor reviews the results, prepares recommendations, communicates with the customer, captures approvals or declines, and moves approved work into execution.

5. What to explore next
Review business capabilities, workflow stages, representative endpoints, and schemas related to findings, recommendations, approvals, and inspection completion.`,

    technical_overview: `1. What it is
The Multi-Point Inspection API is the structured inspection domain for dealership vehicle health checks. It models the inspection lifecycle, findings, media, recommendations, and decision outcomes.

2. Technical model
Technical consumers should think in terms of inspection entities, finding collections, evidence/media attachments, recommendation structures, and approval or completion states.

3. Resource families
Common resource families include inspection records, line-item findings, media or attachments, recommendation artifacts, customer decision objects, and workflow status indicators.

4. Integration concerns
Developers care about how inspection context is created, how findings are represented, how recommendations connect to actionable work, and how approval outcomes are captured for downstream execution.

5. What to explore next
Inspect endpoints, schema groups, object relationships, identifiers, and how inspection data flows into service execution contexts.`,

    architecture_overview: `1. Bounded context role
The Multi-Point Inspection API owns structured inspection and recommendation workflow within the dealership service domain.

2. Boundary
Its responsibility is not full appointment scheduling or full repair-order execution. It sits between intake/service context and downstream execution decisions.

3. Cross-domain relationships
MPI usually depends on upstream intake, visit, or repair-order context and feeds downstream recommendation, approval, and execution processes.

4. Architectural significance
This domain is important because it transforms technician-level observations into customer-facing service decisions while preserving operational traceability.

5. What to explore next
Review relationships between inspection, findings, recommendations, approvals, and execution-oriented downstream systems.`,

    capability_cards: [
      {
        title: 'Start inspection workflow',
        description: 'Begin the MPI process from intake, appointment context, or repair-order workflow.',
        business_guidance: 'This capability launches the inspection lifecycle and provides technicians and advisors a common workflow anchor.',
        advisor_actions: [
          'Initiate inspection from service context',
          'Ensure the inspection is tied to the right vehicle and visit',
          'Use the inspection as the operational anchor for downstream findings'
        ],
        customer_impact: 'Customers benefit from a structured and traceable inspection process.',
        workflow_stage: 'Inspection initiation',
        technical_notes: 'Technical consumers should examine how inspection identity is created, what upstream references are required, and how initialization state is modeled.',
        data_model_notes: 'The core model usually includes inspection identifiers, visit or vehicle references, workflow state, and optional linkage to upstream intake context.',
        integration_notes: 'This capability often bridges appointment, visit, or repair-order context into the inspection domain.',
        prompt: 'Explain how to start the inspection workflow in the multi-point-inspection API'
      },
      {
        title: 'Track inspection progress',
        description: 'Monitor inspection state, technician progress, and workflow status.',
        business_guidance: 'This capability gives advisors visibility into where the inspection stands and when customer-facing follow-up can begin.',
        advisor_actions: [
          'Monitor inspection lifecycle status',
          'Coordinate timing between technician work and advisor communication',
          'Use progress visibility to manage service-lane flow'
        ],
        customer_impact: 'Customers receive more timely updates and better expectation management.',
        workflow_stage: 'Inspection monitoring',
        technical_notes: 'Developers should inspect status models, progress-related fields, and operation families that expose lifecycle transitions.',
        data_model_notes: 'This area often includes workflow status, timestamps, completion indicators, and state transitions.',
        integration_notes: 'Inspection progress is often consumed by advisor tools, dashboards, and customer communication layers.',
        prompt: 'Explain how to track inspection progress in the multi-point-inspection API'
      },
      {
        title: 'Review findings and media',
        description: 'Review technician findings, notes, photos, and condition evidence.',
        business_guidance: 'This capability is where raw technician observations become advisor-usable business information.',
        advisor_actions: [
          'Review condition findings',
          'Use media and notes as supporting evidence',
          'Translate technical observations into customer-facing explanation'
        ],
        customer_impact: 'Customers receive clearer, evidence-based explanations of vehicle condition.',
        workflow_stage: 'Findings review',
        technical_notes: 'Technical teams should examine how findings are represented, how media links are attached, and how condition evidence is structured.',
        data_model_notes: 'The model often includes finding identity, severity or status, notes, media associations, and structured inspection result fields.',
        integration_notes: 'This capability commonly feeds recommendation generation, customer communication, and archival workflows.',
        prompt: 'Explain how to review findings and media in the multi-point-inspection API'
      },
      {
        title: 'Prepare recommendations',
        description: 'Turn findings into advisor-ready repair recommendations and next steps.',
        business_guidance: 'This capability converts inspection results into action and prepares customer-ready proposals tied to vehicle condition and service priorities.',
        advisor_actions: [
          'Turn findings into recommended work',
          'Organize repair guidance by business relevance',
          'Prepare customer-facing next steps'
        ],
        customer_impact: 'Customers receive clearer recommendations and understand why work is being proposed.',
        workflow_stage: 'Recommendation building',
        technical_notes: 'Technical consumers should inspect how recommendations relate to findings, how recommended actions are modeled, and whether downstream execution references exist.',
        data_model_notes: 'This layer often contains structured recommendation entities, finding links, optional pricing context, and approval-ready outputs.',
        integration_notes: 'This capability may connect inspection findings to repair planning, pricing, parts, or service execution contexts.',
        prompt: 'Explain how to prepare recommendations in the multi-point-inspection API'
      },
      {
        title: 'Capture customer approval',
        description: 'Record customer decisions, approvals, and communication outcomes.',
        business_guidance: 'This capability captures the business outcome of the advisor-customer interaction and determines what work moves forward.',
        advisor_actions: [
          'Record approvals and declines',
          'Track customer communication outcomes',
          'Use approval results to determine next workflow step'
        ],
        customer_impact: 'Customers experience a clearer decision process and better documentation of their choices.',
        workflow_stage: 'Decision capture',
        technical_notes: 'Technical teams should inspect approval state, decline/approve semantics, and how decision artifacts are attached to recommendation structures.',
        data_model_notes: 'Common model elements include approval outcomes, timestamps, decision notes, and references to recommendation or finding entities.',
        integration_notes: 'Approval outcomes are often consumed by execution, customer-history, and reporting flows.',
        prompt: 'Explain how customer approval works in the multi-point-inspection API'
      },
      {
        title: 'Close and publish results',
        description: 'Finalize the inspection and share outcomes with internal teams or the customer.',
        business_guidance: 'This capability completes the inspection lifecycle and ensures that finalized outcomes are available for follow-through and communication.',
        advisor_actions: [
          'Finalize inspection outcomes',
          'Publish or share results where needed',
          'Close the loop between inspection, approval, and execution'
        ],
        customer_impact: 'Customers receive a clearer completed inspection story and documented results.',
        workflow_stage: 'Inspection completion',
        technical_notes: 'Technical consumers should inspect completion state, publication or output semantics, and how finalized inspection data is represented.',
        data_model_notes: 'This layer typically includes closed-state indicators, final result artifacts, and references to downstream usage or publication.',
        integration_notes: 'Completion often feeds archival, reporting, customer-history, and execution-oriented systems.',
        prompt: 'Explain how to close and publish results in the multi-point-inspection API'
      }
    ],

    workflow_map: [
      { step: 'Start inspection', detail: 'Launch MPI workflow tied to intake or RO context', prompt: 'Explain how to start the inspection workflow in the multi-point-inspection API' },
      { step: 'Capture findings', detail: 'Technician records results, notes, and media', prompt: 'Explain how findings are captured in the multi-point-inspection API' },
      { step: 'Build recommendations', detail: 'Advisor prepares customer-facing repair guidance', prompt: 'Explain how recommendations are built in the multi-point-inspection API' },
      { step: 'Get approval', detail: 'Customer decisions are captured and tracked', prompt: 'Explain customer approval flow in the multi-point-inspection API' },
      { step: 'Move to execution', detail: 'Approved work moves into repair-order workflow', prompt: 'Explain how approved work moves into execution from the multi-point-inspection API' }
    ],

    schema_cards: [
      { title: 'Inspection', description: 'Core inspection entity representing the inspection lifecycle.', prompt: 'Show schema Inspection in the multi-point-inspection API' },
      { title: 'Finding', description: 'Represents an inspection finding, condition, or observed result.', prompt: 'Show schema Finding in the multi-point-inspection API' },
      { title: 'Recommendation', description: 'Represents advisor-ready or technician-derived recommended work.', prompt: 'Show schema Recommendation in the multi-point-inspection API' },
      { title: 'Approval', description: 'Captures customer approval or decline decisions.', prompt: 'Show schema Approval in the multi-point-inspection API' },
      { title: 'InspectionMedia', description: 'Represents photos, videos, or other evidence attached to findings.', prompt: 'Show schema InspectionMedia in the multi-point-inspection API' },
      { title: 'InspectionStatus', description: 'Represents workflow or lifecycle state of an inspection.', prompt: 'Show schema InspectionStatus in the multi-point-inspection API' },
      { title: 'InspectionManifest', description: 'Defines inspection structure or checklist context.', prompt: 'Show schema InspectionManifest in the multi-point-inspection API' },
      { title: 'InspectionLine', description: 'Represents an inspection line or checklist entry.', prompt: 'Show schema InspectionLine in the multi-point-inspection API' },
      { title: 'ConditionCode', description: 'Represents condition classification or result coding.', prompt: 'Show schema ConditionCode in the multi-point-inspection API' },
      { title: 'MediaReference', description: 'Represents media linkage for inspection evidence.', prompt: 'Show schema MediaReference in the multi-point-inspection API' },
      { title: 'ApprovalDecision', description: 'Represents normalized approval decision values.', prompt: 'Show schema ApprovalDecision in the multi-point-inspection API' },
      { title: 'RecommendationStatus', description: 'Represents recommendation lifecycle or state.', prompt: 'Show schema RecommendationStatus in the multi-point-inspection API' }
    ],

    endpoint_cards: [
      { title: 'GET /inspections', description: 'List or search inspections.', prompt: 'Explain GET /inspections for the multi-point-inspection API' },
      { title: 'POST /inspections', description: 'Create or start an inspection.', prompt: 'Explain POST /inspections for the multi-point-inspection API' },
      { title: 'GET /inspections/{id}', description: 'Retrieve a specific inspection.', prompt: 'Explain GET /inspections/{id} for the multi-point-inspection API' },
      { title: 'GET /inspections/{id}/findings', description: 'Retrieve findings tied to an inspection.', prompt: 'Explain GET /inspections/{id}/findings for the multi-point-inspection API' },
      { title: 'POST /inspections/{id}/recommendations', description: 'Create or record recommendations for an inspection.', prompt: 'Explain POST /inspections/{id}/recommendations for the multi-point-inspection API' },
      { title: 'POST /inspections/{id}/approvals', description: 'Capture approval or decline outcomes.', prompt: 'Explain POST /inspections/{id}/approvals for the multi-point-inspection API' }
    ],

    schema_details: {
      Inspection: {
        title: 'Inspection',
        body: `1. Direct answer
Inspection is the core entity of the Multi-Point Inspection API. It represents the overall inspection lifecycle for a vehicle visit or service event.

2. Key schema details
This schema typically acts as the parent object for inspection status, findings, recommendations, media relationships, and workflow state. Technical consumers should treat it as the anchor entity for the MPI domain.

3. Why it matters
This object is the main entry point for understanding how the inspection is created, tracked, and completed across the service workflow.`
      },
      Finding: {
        title: 'Finding',
        body: `1. Direct answer
Finding represents an observed inspection result, condition, or issue discovered during the inspection process.

2. Key schema details
This schema typically includes the technical observation itself, status or severity information, notes, and relationships to media or recommendations. It is one of the most important technical structures in the MPI domain.

3. Why it matters
Findings are the bridge between technician observation and advisor/customer communication.`
      },
      Recommendation: {
        title: 'Recommendation',
        body: `1. Direct answer
Recommendation represents suggested work or next steps derived from inspection findings.

2. Key schema details
This schema usually links findings to proposed action, service guidance, and downstream approval or execution processes. Technical users should inspect how it references findings and whether it carries pricing, labor, or customer-facing context.

3. Why it matters
Recommendations convert observed condition into actionable service proposals.`
      },
      Approval: {
        title: 'Approval',
        body: `1. Direct answer
Approval captures customer decision outcomes for recommended work.

2. Key schema details
This schema often includes approval state, decision timestamps, notes, and references to recommendations or related workflow objects.

3. Why it matters
Approval is where business intent becomes an execution decision.`
      }
    },

    raw_schemas: {
      Inspection: {
        type: 'object',
        required: ['inspection_id'],
        properties: {
          inspection_id: { type: 'string' },
          inspection_status: { $ref: '#/components/schemas/InspectionStatus' },
          vehicle_reference: { type: 'object' },
          findings: {
            type: 'array',
            items: { $ref: '#/components/schemas/Finding' }
          },
          recommendations: {
            type: 'array',
            items: { $ref: '#/components/schemas/Recommendation' }
          }
        }
      },
      Finding: {
        type: 'object',
        properties: {
          finding_id: { type: 'string' },
          finding_status: { type: 'string' },
          notes: { type: 'string' },
          media: {
            type: 'array',
            items: { $ref: '#/components/schemas/InspectionMedia' }
          }
        }
      },
      Recommendation: {
        type: 'object',
        properties: {
          recommendation_id: { type: 'string' },
          recommendation_text: { type: 'string' },
          related_findings: {
            type: 'array',
            items: { $ref: '#/components/schemas/Finding' }
          }
        }
      },
      Approval: {
        type: 'object',
        properties: {
          approval_id: { type: 'string' },
          approval_status: { type: 'string' },
          decision_timestamp: { type: 'string', format: 'date-time' }
        }
      },
      InspectionMedia: {
        type: 'object',
        properties: {
          media_id: { type: 'string' },
          media_type: { type: 'string' },
          media_url: { type: 'string', format: 'uri' }
        }
      },
      InspectionStatus: {
        type: 'string',
        enum: ['started', 'in_progress', 'completed', 'published']
      }
    },

    endpoint_details: {
      'GET /INSPECTIONS': `1. Direct answer
GET /inspections lists or searches inspection records.

2. Key endpoint details
This endpoint is useful for retrieving inspection collections, filtering workflow state, and supporting advisor or operational views of inspection activity.

3. Why it matters
It gives technical consumers a collection-level entry point into the MPI domain.`,
      'POST /INSPECTIONS': `1. Direct answer
POST /inspections creates or starts an inspection.

2. Key endpoint details
This endpoint is typically used to initialize an inspection record tied to a vehicle visit, intake event, or repair-order context.

3. Why it matters
It is a core lifecycle entry point for the MPI domain.`,
      'GET /INSPECTIONS/{ID}': `1. Direct answer
GET /inspections/{id} retrieves a specific inspection.

2. Key endpoint details
This endpoint is used to inspect the full state of a single inspection, including status, relationships, and downstream context.

3. Why it matters
It is the most direct technical path for reading a specific MPI entity.`,
      'GET /INSPECTIONS/{ID}/FINDINGS': `1. Direct answer
GET /inspections/{id}/findings retrieves findings for a specific inspection.

2. Key endpoint details
This endpoint exposes the technician-observed results and inspection evidence attached to that inspection.

3. Why it matters
It is central to finding review, recommendation generation, and customer communication.`,
      'POST /INSPECTIONS/{ID}/RECOMMENDATIONS': `1. Direct answer
POST /inspections/{id}/recommendations creates or records recommendations for an inspection.

2. Key endpoint details
This endpoint turns findings into proposed next actions or service guidance.

3. Why it matters
It connects technical inspection results to business-ready repair proposals.`,
      'POST /INSPECTIONS/{ID}/APPROVALS': `1. Direct answer
POST /inspections/{id}/approvals records customer approval or decline outcomes.

2. Key endpoint details
This endpoint captures decision state for recommendations or proposed work.

3. Why it matters
It is the bridge from recommendation into execution decision.`
    }
  }
};

function getAudienceOverview(domain, audience) {
  const cfg = DOMAIN_CONFIG[domain];
  if (!cfg) return 'No overview available.';
  if (audience === 'technical') return cfg.technical_overview;
  if (audience === 'architecture') return cfg.architecture_overview;
  return cfg.business_overview;
}

function getBuiltInCapabilityCards(domain) {
  return DOMAIN_CONFIG[domain]?.capability_cards || [];
}

function getBuiltInWorkflowMap(domain) {
  return DOMAIN_CONFIG[domain]?.workflow_map || [];
}

function getBuiltInSchemaCards(domain) {
  return DOMAIN_CONFIG[domain]?.schema_cards || [];
}

function getBuiltInEndpointCards(domain) {
  return DOMAIN_CONFIG[domain]?.endpoint_cards || [];
}

function getBuiltInSchemaDetails(domain, schemaName) {
  const normalized = String(schemaName || '').trim();
  return DOMAIN_CONFIG[domain]?.schema_details?.[normalized] || null;
}

function getBuiltInRawSchema(domain, schemaName) {
  const normalized = String(schemaName || '').trim();
  return DOMAIN_CONFIG[domain]?.raw_schemas?.[normalized] || null;
}

function getBuiltInEndpointDetail(domain, endpointTitle) {
  return DOMAIN_CONFIG[domain]?.endpoint_details?.[normalizeEndpointTitle(endpointTitle)] || null;
}

/* --------------------------------------------------
 * SAFE MCP ENRICHMENT
 * -------------------------------------------------- */

async function safeListSchemas(domain, limit = 6) {
  try {
    return await toolCached('listSchemas', { domain_name: domain, limit });
  } catch {
    return { schemas: [] };
  }
}

async function safeListOperations(domain, limit = 6) {
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

  let intro;
  if (audience === 'technical') {
    intro = `1. Direct answer
Technical capability views for the ${domain} API are shown below.

2. Key details
Each capability includes technical notes, data model notes, and integration notes in addition to business meaning.

3. Useful next steps
Select a capability card to inspect technical details and implementation context.`;
  } else if (audience === 'architecture') {
    intro = `1. Direct answer
Architecture-oriented capability views for the ${domain} API are shown below.

2. Key details
Each capability includes context boundaries, data-model implications, and integration implications relevant to architecture work.

3. Useful next steps
Select a capability card to inspect architecture-oriented details and dependencies.`;
  } else {
    intro = `1. Direct answer
Business capabilities for the ${domain} API are shown below.

2. Key details
These capabilities are organized around the dealership workflow and include advisor-facing meaning, workflow stage, and customer impact.

3. Useful next steps
Select a capability card to go deeper into business guidance and next actions.`;
  }

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

MCP overview:
${JSON.stringify(overview)}

Operations:
${JSON.stringify(ops)}

Schemas:
${JSON.stringify(schemas)}

Write a rich structured answer that preserves the depth of the built-in domain overview while using MCP data to enrich it where possible.`
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

Live overview:
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

  return {
    answer,
    sections: extractSections(answer),
    audience,
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: deriveEndpointCards(ops, domain).length ? deriveEndpointCards(ops, domain) : getBuiltInEndpointCards(domain),
    schema_cards: deriveSchemaCards(schemas, domain).length ? deriveSchemaCards(schemas, domain) : getBuiltInSchemaCards(domain),
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

  try {
    liveSchema = await toolCached('getSchema', {
      domain_name: domain,
      schema_name: schemaName
    });
  } catch {
    liveSchema = null;
  }

  const builtIn = getBuiltInSchemaDetails(domain, schemaName);
  const rawSchema = liveSchema || getBuiltInRawSchema(domain, schemaName);

  let answer;
  if (liveSchema) {
    try {
      const response = await withTimeout(
        client.responses.create({
          model: MODEL,
          input: `You are a STAR automotive retail API assistant.
Audience: ${audience}

Schema name:
${schemaName}

Schema data:
${JSON.stringify(liveSchema)}

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
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: getBuiltInEndpointCards(domain),
    schema_cards: getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: !liveSchema
  };
}

async function buildEndpointResponse(domain, message, audience) {
  const ops = await safeListOperations(domain, 6);
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
  const builtIn = getBuiltInEndpointDetail(domain, endpointTitle);

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
    capability_cards: getBuiltInCapabilityCards(domain),
    endpoint_cards: getBuiltInEndpointCards(domain),
    schema_cards: getBuiltInSchemaCards(domain),
    workflow_map: getBuiltInWorkflowMap(domain),
    progressive: true
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

    if (!MCP_SERVER_URL) {
      return res.status(500).json({ error: 'Missing MCP_SERVER_URL' });
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
