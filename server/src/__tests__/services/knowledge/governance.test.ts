import { describe, it, expect, beforeEach } from 'vitest';
import { initDb, getDb } from '../../../db/index.js';
import { parsePolicy, redactText, isModelAllowed, isContentTypeAllowed, audit, listAudit, sweepRetention, setPolicyForTests, getPolicy } from '../../../services/knowledge/governance.js';

const minimal = parsePolicy('version: 1\nname: t\n');

describe('governance policy parsing', () => {
  it('fills defaults and versions by hash', () => {
    expect(minimal.ingest.redact_pii).toBe(true);
    expect(minimal.retrieval.max_chunks).toBe(12);
    expect(minimal.generation.refuse_without_evidence).toBe(true);
    expect(minimal.graph.extractions_per_minute).toBe(20);
    expect(minimal.generation.wait_for_reset_seconds).toBe(0);
    expect(minimal.versionTag).toMatch(/^t@1#[0-9a-f]{12}$/);
  });

  it('rejects invalid values with a configuration error', () => {
    expect(() => parsePolicy('version: 1\nname: t\nretrieval:\n  max_chunks: -1\n')).toThrow(/governance schema error/);
    expect(() => parsePolicy('version: 1\nname: t\ningest:\n  custom_redactions:\n    - name: bad\n      pattern: "("\n')).toThrow(/invalid pattern/);
  });

  it('parses the shipped governance.yaml', () => {
    setPolicyForTests(null);
    const p = getPolicy();
    expect(p.name).toBe('wall-e-default');
    expect(p.graph.extract).toBe(true);
  });
});

describe('PII redaction', () => {
  it('redacts emails, phones, Luhn-valid cards, IBANs and API keys', () => {
    const p = parsePolicy('version: 1\nname: t\ningest:\n  pii_detectors: [email, phone, credit_card, iban, api_key, ssn, ip_address]\n');
    const r = redactText([
      'Contact jane.doe@example.com or +1 (415) 555-0134.',
      'Card 4111 1111 1111 1111 works, order 12345678 does not.',
      'IBAN DE89 3704 0044 0532 0130 00 and key gsk_P4LElOmLECuNh9jpAS4RWGdyb3FYoKEdWVxiaFFuqtfD.',
      'Version 1.2.3 released 2026-09-09 on host 10.0.0.12 with ssn 123-45-6789.',
    ].join('\n'), p);
    expect(r.text).toContain('[REDACTED:email]');
    expect(r.text).toContain('[REDACTED:phone]');
    expect(r.text).toContain('[REDACTED:credit_card] works');
    expect(r.text).toContain('order 12345678 does not');
    expect(r.text).toContain('[REDACTED:iban]');
    expect(r.text).toContain('[REDACTED:api_key]');
    expect(r.text).toContain('[REDACTED:ip_address]');
    expect(r.text).toContain('[REDACTED:ssn]');
    expect(r.text).toContain('Version 1.2.3 released 2026-09-09');
    expect(r.total).toBe(7);
    expect(r.redactions.find(x => x.detector === 'email')?.count).toBe(1);
  });

  it('applies named custom redactions and can be switched off', () => {
    const p = parsePolicy('version: 1\nname: t\ningest:\n  pii_detectors: []\n  custom_redactions:\n    - name: employee_id\n      pattern: "EMP-[0-9]{6}"\n');
    expect(redactText('badge EMP-123456 ok', p).text).toBe('badge [REDACTED:employee_id] ok');
    const off = parsePolicy('version: 1\nname: t\ningest:\n  redact_pii: false\n');
    expect(redactText('mail me@x.io', off).text).toBe('mail me@x.io');
  });
});

describe('model and content-type gates', () => {
  it('matches platform/, platform/model and bare model rules; deny beats allow', () => {
    const p = parsePolicy('version: 1\nname: t\ngeneration:\n  allowed_models: ["groq/", "openai/gpt-4o", "llama-3.3-70b"]\n  denied_models: ["groq/bad-model"]\n');
    expect(isModelAllowed('groq', 'llama-3.1-8b', p)).toBe(true);
    expect(isModelAllowed('groq', 'bad-model', p)).toBe(false);
    expect(isModelAllowed('openai', 'gpt-4o', p)).toBe(true);
    expect(isModelAllowed('openai', 'gpt-4o-mini', p)).toBe(false);
    expect(isModelAllowed('cerebras', 'llama-3.3-70b', p)).toBe(true);
    expect(isModelAllowed('anything', 'x', minimal)).toBe(true);
  });

  it('checks content types case-insensitively and ignores parameters', () => {
    expect(isContentTypeAllowed('Text/Markdown; charset=utf-8', minimal)).toBe(true);
    expect(isContentTypeAllowed('application/pdf', minimal)).toBe(false);
  });
});

describe('audit log and retention', () => {
  beforeEach(() => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    initDb(':memory:');
  });

  it('writes and lists audit rows with the policy version', () => {
    audit({ action: 'ingest', actor: 'user:1', baseId: 7, target: 'document:3', details: { chunks: 2 } }, minimal);
    audit({ action: 'query', actor: 'unified', baseId: 7, target: 'query:x' }, minimal);
    const rows = listAudit({ baseId: 7 });
    expect(rows).toHaveLength(2);
    expect(rows[0].action).toBe('query');
    expect(rows[1].policy_version).toBe(minimal.versionTag);
    expect(JSON.parse(rows[1].details_json)).toEqual({ chunks: 2 });
    expect(listAudit({ action: 'ingest' })).toHaveLength(1);
  });

  it('honours audit.enabled=false', () => {
    const off = parsePolicy('version: 1\nname: t\naudit:\n  enabled: false\n');
    audit({ action: 'ingest' }, off);
    expect(listAudit()).toHaveLength(0);
  });

  it('sweeps old provenance and audit rows and caps eval runs', () => {
    const db = getDb();
    const dayMs = 86_400_000;
    const now = Date.now();
    db.prepare('INSERT INTO knowledge_bases (slug, name, created_at_ms, updated_at_ms) VALUES (?, ?, ?, ?)').run('b', 'b', now, now);
    const insQ = db.prepare("INSERT INTO knowledge_queries (id, base_id, question, question_hash, created_at_ms) VALUES (?, 1, 'q', 'h', ?)");
    insQ.run('old', now - 400 * dayMs);
    insQ.run('new', now);
    db.prepare("INSERT INTO knowledge_audit (at_ms, action) VALUES (?, 'x')").run(now - 800 * dayMs);
    db.prepare("INSERT INTO knowledge_audit (at_ms, action) VALUES (?, 'x')").run(now);
    const insE = db.prepare("INSERT INTO knowledge_eval_runs (id, base_id, dataset, started_at_ms) VALUES (?, 1, 'd', ?)");
    for (let i = 0; i < 5; i++) insE.run(`e${i}`, now - i);
    const p = parsePolicy('version: 1\nname: t\nretention:\n  provenance_days: 365\n  audit_days: 730\n  eval_runs_keep: 3\n');
    const swept = sweepRetention(p, now);
    expect(swept).toEqual({ queries: 1, audit: 1, evalRuns: 2 });
    expect((db.prepare('SELECT COUNT(*) n FROM knowledge_queries').get() as any).n).toBe(1);
    expect((db.prepare('SELECT id FROM knowledge_eval_runs ORDER BY started_at_ms DESC').all() as any[]).map(r => r.id)).toEqual(['e0', 'e1', 'e2']);
  });
});
