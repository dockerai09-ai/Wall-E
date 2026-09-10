import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { initDb } from '../../../db/index.js';
import { loadLoopConfig, runLoop, guardrailFailures, objectiveOf, resolveTarget } from '../../../services/stt/loop.js';
import { insertRun } from '../../../services/stt/store.js';
import { variantKey, type SttVariant } from '../../../services/stt/pipeline.js';
import type { BenchRun } from '../../../services/stt/bench.js';
import type { AggregateMetrics } from '../../../services/stt/metrics.js';

function metrics(over: Partial<AggregateMetrics>): AggregateMetrics {
  return { cases: 10, scored: 10, errors: 0, wer: 0.1, cer: 0.05, werMean: 0.1, termRecall: 0.95, piiLeakRate: 0, latencyP50Ms: 500, latencyP95Ms: 900, correctionRejections: 0, ...over };
}

/** A fake benchmark: the WER of a variant is a sum of per-knob penalties. */
function fakeRun(v: SttVariant, table: Record<string, number>): BenchRun {
  const w = 0.02 + (table[`model:${v.sttModel}`] ?? 0) + (table[`prompt:${v.prompt}`] ?? 0) + (table[`corr:${v.correction}`] ?? 0) + (table[`cm:${v.correctionModel}`] ?? 0);
  return { profile: 'sv-medical', arm: 'wall-e', variant: v, glossaryHash: 'x', metrics: metrics({ wer: w, werMean: w }), cases: [], startedAt: 0, finishedAt: 0, misses: [] };
}

describe('stt loop', () => {
  let outDir: string;
  beforeEach(() => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    initDb(':memory:');
    outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'stt-loop-'));
  });

  it('loads loop.yaml with defaults and evaluates guardrails and objectives', () => {
    const c = loadLoopConfig('sv-medical');
    expect(c.objective).toBe('wer');
    expect(c.search.stt_model.length).toBeGreaterThan(0);
    expect(guardrailFailures(metrics({ termRecall: 0.5, piiLeakRate: 0.1 }), c.guardrails)).toHaveLength(2);
    expect(guardrailFailures(metrics({ cases: 45, scored: 14, errors: 31 }), c.guardrails)[0]).toMatch(/error rate/);
    expect(guardrailFailures(metrics({}), c.guardrails)).toEqual([]);
    expect(objectiveOf(metrics({ termRecall: 0.9 }), 'term_error')).toBeCloseTo(0.1);
    expect(objectiveOf(metrics({ cer: 0.03 }), 'cer')).toBe(0.03);
  });

  it('takes the target from the latest imported arm run, else from the config', () => {
    const c = loadLoopConfig('sv-medical');
    expect(resolveTarget(c)).toMatchObject({ source: 'config', value: c.target.wer });
    insertRun({ kind: 'bench', profile: 'sv-medical', arm: 'claude-pro', loop_id: null, iteration: null, variant_json: 'null', metrics_json: JSON.stringify(metrics({ wer: 0.2 })), cases_json: '[]', status: 'done' });
    expect(resolveTarget(c)).toMatchObject({ source: 'arm', value: 0.2, arm: 'claude-pro' });
  });

  it('descends over the search space, adopts improvements, and stops when the target is beaten', async () => {
    const c = loadLoopConfig('sv-medical');
    c.learn.enabled = false;
    c.target.arm = null;
    c.target.wer = 0.05;
    c.target.margin = 0.0;
    c.start = { sttModel: 'whisper-large-v3-turbo', prompt: 'none', correction: 'off', correctionModel: null, temperature: 0 };
    c.search = { stt_model: ['whisper-large-v3-turbo', 'whisper-large-v3'], prompt: ['glossary', 'none'], correction: ['llm', 'off'], correction_model: [null, 'groq/big'], temperature: [0] };
    const table = { 'prompt:none': 0.04, 'corr:off': 0.03, 'model:whisper-large-v3': 0.01, 'cm:groq/big': -0.005 };
    const evaluated: string[] = [];
    const state = await runLoop(c, {
      cases: [], outDir, persist: false,
      evaluate: async v => { evaluated.push(variantKey(v)); return fakeRun(v, table); },
    });
    expect(state.status).toBe('target_met');
    expect(state.best?.variant).toMatchObject({ sttModel: 'whisper-large-v3-turbo', prompt: 'glossary', correction: 'llm', correctionModel: 'groq/big' });
    expect(state.best?.objective).toBeCloseTo(0.015);
    // the incumbent is never re-evaluated and the report is on disk
    expect(new Set(evaluated).size).toBe(evaluated.length);
    expect(fs.existsSync(path.join(outDir, 'state.json'))).toBe(true);
    expect(fs.readFileSync(path.join(outDir, 'report.md'), 'utf8')).toContain('target_met');
  });

  it('converges when nothing improves and reports guardrail failures', async () => {
    const c = loadLoopConfig('sv-medical');
    c.learn.enabled = false;
    c.target.arm = null;
    c.target.wer = 0.0;
    c.budget.patience = 1;
    c.start = { sttModel: 'a', prompt: 'glossary', correction: 'llm', correctionModel: null, temperature: 0 };
    c.search = { stt_model: ['a', 'b'], prompt: ['glossary'], correction: ['llm'], correction_model: [null], temperature: [0] };
    const state = await runLoop(c, {
      cases: [], outDir, persist: false,
      evaluate: async v => {
        const run = fakeRun(v, { 'model:b': -0.01 });
        if (v.sttModel === 'b') run.metrics.piiLeakRate = 0.5; // better WER but leaks
        return run;
      },
    });
    expect(state.status).toBe('converged');
    expect(state.best?.variant.sttModel).toBe('a');
    expect(state.evaluations.find(e => e.variant.sttModel === 'b')?.guardrails[0]).toMatch(/PII leak/);
  });
});
