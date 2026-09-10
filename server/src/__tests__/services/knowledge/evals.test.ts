import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { ChatMessage } from '@freellmapi/shared/types.js';
import { initDb } from '../../../db/index.js';
import { ingestDocument } from '../../../services/knowledge/ingest.js';
import { runIndexerOnce } from '../../../services/knowledge/indexer.js';
import { setKnowledgeChatForTests } from '../../../services/knowledge/llm.js';
import { parsePolicy, setPolicyForTests } from '../../../services/knowledge/governance.js';
import { setGraphMirrorForTests } from '../../../services/knowledge/neo4j.js';
import { parseJsonl, runEval, evalReportMarkdown, sourceMatches } from '../../../services/knowledge/evals.js';
import * as store from '../../../services/knowledge/store.js';

const chat = async (messages: ChatMessage[]) => {
  const system = String(messages[0].content);
  if (system.includes('strict grader')) {
    return { text: '{"correctness": 0.9, "faithfulness": 1.0, "reasoning": "matches the reference"}', platform: 'test', modelId: 'judge', usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } };
  }
  return { text: 'Port 3001 by default [1].', platform: 'test', modelId: 'answerer', usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } };
};

describe('knowledge evals', () => {
  let base: store.BaseRow;

  beforeEach(async () => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    initDb(':memory:');
    setPolicyForTests(parsePolicy('version: 1\nname: t\ngraph:\n  extract: false\n'));
    setGraphMirrorForTests(null);
    setKnowledgeChatForTests(chat);
    base = store.createBase({ slug: 'docs', name: 'Docs', embedder: 'hash', embeddingModel: 'hash-256' });
    await ingestDocument(base, { title: 'Install', text: 'The dashboard and API listen on port 3001 by default. Set PORT to change it.', source: 'docs/en/install/01-install.md', actor: 'u' });
    await ingestDocument(base, { title: 'Proxy', text: 'PROXY_URL points outbound requests at a SOCKS or HTTP proxy.', source: 'docs/en/env/03-outbound-proxies.md', actor: 'u' });
    await runIndexerOnce();
  });

  afterEach(() => {
    setKnowledgeChatForTests(null);
    setPolicyForTests(null);
    setGraphMirrorForTests(undefined);
  });

  it('parses JSONL datasets and validates cases', () => {
    const cases = parseJsonl('{"question":"a"}\n# comment\n\n{"id":"x","question":"b","expected_sources":["s"]}\n');
    expect(cases.map(c => c.id)).toEqual(['case-1', 'x']);
    expect(() => parseJsonl('{"nope":1}')).toThrow(/line 1/);
    expect(() => parseJsonl('not json')).toThrow(/not valid JSON/);
    expect(() => parseJsonl('')).toThrow(/no cases/);
  });

  it('matches sources by path suffix or title', () => {
    expect(sourceMatches({ source: 'docs/en/install/01-install.md', title: 'Install' }, '01-install.md')).toBe(true);
    expect(sourceMatches({ source: '', title: 'Install' }, 'install')).toBe(true);
    expect(sourceMatches({ source: 'a/b.md', title: 'x' }, 'c.md')).toBe(false);
  });

  it('runs a dataset, scores retrieval and answers, and renders a report', async () => {
    const cases = parseJsonl([
      JSON.stringify({ id: 'port', question: 'Which port does the dashboard listen on?', expected_answer: '3001', expected_sources: ['docs/en/install/01-install.md'], tags: ['install'] }),
      JSON.stringify({ id: 'miss', question: 'Which port does the dashboard listen on?', expected_sources: ['docs/en/does-not-exist.md'] }),
    ].join('\n'));
    const run = await runEval(base, cases, { datasetName: 'unit', actor: 'u', judge: true, k: 3 });
    expect(run.status).toBe('done');
    expect(run.cases).toBe(2);
    const metrics = JSON.parse(run.metrics_json);
    expect(metrics.hitRate).toBe(0.5);
    expect(metrics.recallAtK).toBe(0.5);
    expect(metrics.mrr).toBe(0.5);
    expect(metrics.correctness).toBe(0.9);
    expect(metrics.faithfulness).toBe(1);
    expect(metrics.judged).toBe(2);
    expect(metrics.citedRate).toBe(1);
    expect(metrics.refusalRate).toBe(0);
    expect(metrics.graph.entities).toBe(0);
    const results = JSON.parse(run.results_json);
    expect(results[0].hit).toBe(true);
    expect(results[0].citationPrecision).toBe(1);
    expect(results[0].queryId).toBeTruthy();
    expect(results[1].hit).toBe(false);
    expect(results[1].matchedSources).toEqual([]);

    const md = evalReportMarkdown(run, base.slug);
    expect(md).toContain('# Knowledge eval — unit on `docs`');
    expect(md).toContain('| Retrieval hit rate | 50.0% |');
    expect(md).toContain('## Needs attention');
    expect(md).toContain('**miss**');
    expect(store.listEvalRuns(base.id)[0].id).toBe(run.id);
    expect(store.listEvalRuns(null)[0].results_json).toBe('[]');
  });

  it('records per-case errors without failing the run', async () => {
    setKnowledgeChatForTests(async () => { throw new Error('boom'); });
    const run = await runEval(base, parseJsonl('{"id":"e","question":"Which port does the dashboard listen on?","expected_sources":["01-install.md"]}'), { datasetName: 'err', actor: 'u' });
    expect(run.status).toBe('done');
    const metrics = JSON.parse(run.metrics_json);
    expect(metrics.errors).toBe(1);
    expect(metrics.errorRate).toBe(1);
    // Retrieval is still scored when generation fails.
    expect(metrics.hitRate).toBe(1);
    const r = JSON.parse(run.results_json)[0];
    expect(r.error).toContain('boom');
    expect(r.hit).toBe(true);
    expect(r.retrieved.length).toBeGreaterThan(0);
  });
});
