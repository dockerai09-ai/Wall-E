// Governance policy: what may be ingested, retrieved and generated, and the
// audit trail of every enforcement. Loaded from knowledge/governance.yaml
// (KB_GOVERNANCE_PATH), validated with zod, cached, versioned by hash so an
// audit row can name the exact policy that fired.

import fs from 'fs';
import crypto from 'crypto';
import { parse as parseYaml } from 'yaml';
import { z } from 'zod';
import { getDb } from '../../db/index.js';
import { getKnowledgeConfig, KnowledgeError } from './config.js';

const BUILTIN_DETECTORS = ['email', 'phone', 'credit_card', 'iban', 'ssn', 'personnummer', 'api_key', 'ip_address'] as const;
export type PiiDetector = typeof BUILTIN_DETECTORS[number];

const policySchema = z.object({
  version: z.number().int().nonnegative(),
  name: z.string().min(1),
  ingest: z.object({
    redact_pii: z.boolean().default(true),
    pii_detectors: z.array(z.enum(BUILTIN_DETECTORS)).default(['email', 'phone', 'credit_card', 'iban', 'api_key']),
    custom_redactions: z.array(z.object({ name: z.string().min(1), pattern: z.string().min(1) })).default([]),
    max_document_bytes: z.number().int().positive().default(10 * 1024 * 1024),
    allowed_content_types: z.array(z.string()).default(['text/plain', 'text/markdown', 'text/html', 'application/json', 'text/csv']),
    dedupe_by_content_hash: z.boolean().default(true),
    chunk: z.object({
      target_tokens: z.number().int().positive().default(400),
      overlap_tokens: z.number().int().nonnegative().default(60),
    }).default({}),
  }).default({}),
  retrieval: z.object({
    max_chunks: z.number().int().positive().default(12),
    min_score: z.number().min(0).default(0.05),
    hybrid: z.boolean().default(true),
    graph: z.object({
      enabled: z.boolean().default(true),
      max_hops: z.number().int().min(1).max(4).default(2),
      max_entities: z.number().int().positive().default(8),
      max_context_chunks: z.number().int().nonnegative().default(6),
    }).default({}),
  }).default({}),
  generation: z.object({
    require_citations: z.boolean().default(true),
    refuse_without_evidence: z.boolean().default(true),
    allowed_models: z.array(z.string()).default([]),
    denied_models: z.array(z.string()).default([]),
    max_context_tokens: z.number().int().positive().default(6000),
    temperature: z.number().min(0).max(2).default(0.2),
    // When every route is rate-limited, wait up to this long for the router's
    // announced reset before failing. 0 = fail immediately (API clients retry).
    wait_for_reset_seconds: z.number().int().min(0).max(300).default(0),
  }).default({}),
  graph: z.object({
    extract: z.boolean().default(true),
    extraction_model: z.string().nullable().default(null),
    min_confidence: z.number().min(0).max(1).default(0.5),
    catch_all_warn_ratio: z.number().min(0).max(1).default(0.4),
    // Free tiers are shared with the answers; a burst of extraction calls
    // can bench every model for a minute. 20/min = one chunk per indexer tick.
    extractions_per_minute: z.number().int().min(1).max(600).default(20),
  }).default({}),
  retention: z.object({
    provenance_days: z.number().int().nonnegative().default(365),
    audit_days: z.number().int().nonnegative().default(730),
    eval_runs_keep: z.number().int().nonnegative().default(200),
  }).default({}),
  audit: z.object({
    enabled: z.boolean().default(true),
    log_to_console: z.boolean().default(false),
  }).default({}),
  access: z.object({
    scope_by_profile: z.boolean().default(true),
    allow_delete: z.boolean().default(true),
  }).default({}),
});

export type PolicyDoc = z.infer<typeof policySchema>;

export interface Policy extends PolicyDoc {
  /** name@version#hash — what audit rows and provenance record. */
  versionTag: string;
  hash: string;
  path: string;
  loadedAtMs: number;
  customRedactions: { name: string; re: RegExp }[];
}

let cached: Policy | null = null;

function finish(doc: PolicyDoc, path: string): Policy {
  const canonical = JSON.stringify(doc);
  const hash = crypto.createHash('sha256').update(canonical).digest('hex').slice(0, 12);
  const customRedactions = doc.ingest.custom_redactions.map(r => {
    try {
      return { name: r.name, re: new RegExp(r.pattern, 'g') };
    } catch (err: any) {
      throw new KnowledgeError(`governance custom_redactions.${r.name} has an invalid pattern: ${err?.message ?? err}`, 500, 'configuration_error');
    }
  });
  return { ...doc, versionTag: `${doc.name}@${doc.version}#${hash}`, hash, path, loadedAtMs: Date.now(), customRedactions };
}

export function parsePolicy(yamlText: string, path = '<inline>'): Policy {
  let raw: unknown;
  try {
    raw = parseYaml(yamlText);
  } catch (err: any) {
    throw new KnowledgeError(`governance YAML is invalid: ${err?.message ?? err}`, 500, 'configuration_error');
  }
  const parsed = policySchema.safeParse(raw ?? {});
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    throw new KnowledgeError(`governance schema error at ${issue.path.join('.') || '<root>'}: ${issue.message}`, 500, 'configuration_error');
  }
  return finish(parsed.data, path);
}

export function loadPolicy(path = getKnowledgeConfig().governancePath): Policy {
  if (!fs.existsSync(path)) {
    console.warn(`[knowledge] governance file not found at ${path}; using built-in defaults`);
    return finish(policySchema.parse({ version: 0, name: 'builtin-defaults' }), '<builtin>');
  }
  return parsePolicy(fs.readFileSync(path, 'utf8'), path);
}

export function getPolicy(): Policy {
  if (!cached) cached = loadPolicy();
  return cached;
}

export function reloadPolicy(): Policy {
  cached = loadPolicy();
  return cached;
}

export function setPolicyForTests(p: Policy | null): void {
  cached = p;
}

// ------------------------------------------------------------- redaction ----

export function luhnOk(digits: string): boolean {
  let sum = 0;
  let alt = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let d = digits.charCodeAt(i) - 48;
    if (alt) { d *= 2; if (d > 9) d -= 9; }
    sum += d;
    alt = !alt;
  }
  return sum % 10 === 0;
}

// Each detector returns the replacement or null to leave the match alone,
// so the regexes can stay loose and the callback carries the precision.
const DETECTORS: Record<PiiDetector, { re: RegExp; keep?: (m: string) => boolean }> = {
  email: { re: /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g },
  phone: {
    re: /(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b/g,
    // 9-15 digits, and a bare run of 12+ digits with no separator is far more
    // likely an id or account number than a phone number.
    keep: m => { const n = m.replace(/\D/g, '').length; return n >= 9 && n <= 15 && (n <= 11 || /[\s.()+-]/.test(m)); },
  },
  credit_card: {
    // Ends on a digit so the replacement never eats the following space.
    re: /\b\d(?:[ -]?\d){12,18}\b/g,
    keep: m => { const d = m.replace(/\D/g, ''); return d.length >= 13 && d.length <= 19 && luhnOk(d); },
  },
  iban: { re: /\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,4})?\b/g },
  ssn: { re: /\b\d{3}-\d{2}-\d{4}\b/g },
  // Swedish personal identity number: YYMMDD-NNNC or YYYYMMDD-NNNC; the
  // separator may be '-', '+' (persons over 100), a space (as a speech
  // recogniser writes it) or absent. The last digit is a Luhn check over the
  // ten-digit form, which keeps ordinary ten/twelve-digit numbers out.
  personnummer: {
    re: /\b(?:19|20)?\d{6}\s?[-+]?\s?\d{4}\b/g,
    keep: m => {
      const ten = m.replace(/\D/g, '').slice(-10);
      const month = Number(ten.slice(2, 4));
      const day = Number(ten.slice(4, 6));
      // day 61-91 = samordningsnummer (coordination number)
      return month >= 1 && month <= 12 && day >= 1 && day <= 91 && luhnOk(ten);
    },
  },
  api_key: {
    re: /\b(?:sk|gsk|ghp|gho|xox[bap]|AKIA|AIza|hf)_?[A-Za-z0-9_-]{16,}\b|(?:api[_-]?key|token|secret|password)\s*[:=]\s*["']?[A-Za-z0-9_\-./+]{12,}/gi,
  },
  ip_address: {
    re: /\b(?:\d{1,3}\.){3}\d{1,3}\b/g,
    keep: m => m.split('.').every(o => Number(o) <= 255),
  },
};

const DETECTOR_ORDER: PiiDetector[] = ['credit_card', 'iban', 'personnummer', 'ssn', 'api_key', 'email', 'ip_address', 'phone'];

export interface RedactionReport {
  text: string;
  redactions: { detector: string; count: number }[];
  total: number;
}

export function redactText(text: string, policy: Policy = getPolicy()): RedactionReport {
  if (!policy.ingest.redact_pii) return { text, redactions: [], total: 0 };
  let out = text;
  const counts = new Map<string, number>();
  const apply = (name: string, re: RegExp, keep?: (m: string) => boolean) => {
    out = out.replace(re, m => {
      if (keep && !keep(m)) return m;
      counts.set(name, (counts.get(name) ?? 0) + 1);
      return `[REDACTED:${name}]`;
    });
  };
  // Fixed precedence, most specific first, whatever order the policy lists:
  // the loose phone pattern would otherwise swallow card and IBAN digits.
  const enabled = new Set(policy.ingest.pii_detectors);
  for (const name of DETECTOR_ORDER) {
    if (!enabled.has(name)) continue;
    const d = DETECTORS[name];
    apply(name, d.re, d.keep);
  }
  for (const c of policy.customRedactions) apply(c.name, c.re);
  const redactions = [...counts.entries()].map(([detector, count]) => ({ detector, count }));
  return { text: out, redactions, total: redactions.reduce((n, r) => n + r.count, 0) };
}

// ------------------------------------------------------------ model gate ----

function matchesModelRule(rule: string, platform: string, modelId: string): boolean {
  const r = rule.toLowerCase();
  const p = platform.toLowerCase();
  const m = modelId.toLowerCase();
  if (r.endsWith('/')) return p === r.slice(0, -1);
  if (r.includes('/')) return r === `${p}/${m}`;
  return r === m || r === p;
}

export function isModelAllowed(platform: string, modelId: string, policy: Policy = getPolicy()): boolean {
  const { allowed_models, denied_models } = policy.generation;
  if (denied_models.some(rule => matchesModelRule(rule, platform, modelId))) return false;
  if (allowed_models.length === 0) return true;
  return allowed_models.some(rule => matchesModelRule(rule, platform, modelId));
}

export function isContentTypeAllowed(contentType: string, policy: Policy = getPolicy()): boolean {
  const ct = contentType.split(';')[0].trim().toLowerCase();
  return policy.ingest.allowed_content_types.map(t => t.toLowerCase()).includes(ct);
}

// ----------------------------------------------------------------- audit ----

export interface AuditEntry {
  action: string;
  actor?: string;
  baseId?: number | null;
  target?: string;
  details?: Record<string, unknown>;
}

export function audit(entry: AuditEntry, policy: Policy = getPolicy()): void {
  if (!policy.audit.enabled) return;
  try {
    getDb().prepare(`
      INSERT INTO knowledge_audit (at_ms, actor, action, base_id, target, details_json, policy_version)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).run(Date.now(), entry.actor ?? '', entry.action, entry.baseId ?? null, entry.target ?? '', JSON.stringify(entry.details ?? {}), policy.versionTag);
    if (policy.audit.log_to_console) {
      console.log(`[knowledge/audit] ${entry.action} ${entry.target ?? ''} by ${entry.actor ?? '-'} ${JSON.stringify(entry.details ?? {})}`);
    }
  } catch (err: any) {
    console.error(`[knowledge/audit] failed to write audit row: ${err?.message ?? err}`);
  }
}

export interface AuditRow {
  id: number;
  at_ms: number;
  actor: string;
  action: string;
  base_id: number | null;
  target: string;
  details_json: string;
  policy_version: string;
}

export function listAudit(opts: { baseId?: number; limit?: number; action?: string } = {}): AuditRow[] {
  const clauses: string[] = [];
  const params: unknown[] = [];
  if (opts.baseId != null) { clauses.push('base_id = ?'); params.push(opts.baseId); }
  if (opts.action) { clauses.push('action = ?'); params.push(opts.action); }
  const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : '';
  params.push(Math.min(Math.max(opts.limit ?? 100, 1), 1000));
  return getDb().prepare(`SELECT * FROM knowledge_audit ${where} ORDER BY at_ms DESC, id DESC LIMIT ?`).all(...params) as AuditRow[];
}

/** Apply retention: old provenance and audit rows go, eval runs are capped. */
export function sweepRetention(policy: Policy = getPolicy(), now = Date.now()): { queries: number; audit: number; evalRuns: number } {
  const db = getDb();
  const dayMs = 24 * 60 * 60 * 1000;
  let queries = 0;
  let auditRows = 0;
  let evalRuns = 0;
  if (policy.retention.provenance_days > 0) {
    queries = db.prepare('DELETE FROM knowledge_queries WHERE created_at_ms < ?').run(now - policy.retention.provenance_days * dayMs).changes;
  }
  if (policy.retention.audit_days > 0) {
    auditRows = db.prepare('DELETE FROM knowledge_audit WHERE at_ms < ?').run(now - policy.retention.audit_days * dayMs).changes;
  }
  if (policy.retention.eval_runs_keep > 0) {
    evalRuns = db.prepare(`
      DELETE FROM knowledge_eval_runs WHERE id IN (
        SELECT id FROM knowledge_eval_runs ORDER BY started_at_ms DESC LIMIT -1 OFFSET ?
      )
    `).run(policy.retention.eval_runs_keep).changes;
  }
  return { queries, audit: auditRows, evalRuns };
}

export function policySummary(p: Policy) {
  const { versionTag, hash, path, loadedAtMs, customRedactions: _c, ...doc } = p;
  return { versionTag, hash, path, loadedAt: new Date(loadedAtMs).toISOString(), policy: doc };
}
