// Background indexer: embeds pending chunks in batches, runs graph extraction
// (one model call per chunk, bounded per tick), flips documents to 'ready',
// and sweeps retention. Started from index.ts onReady like the other jobs;
// tests call runIndexerOnce() directly.

import type { Scheduler } from '../../lib/scheduler.js';
import { getDb } from '../../db/index.js';
import { embedderFor } from './embedder.js';
import { getPolicy, sweepRetention } from './governance.js';
import { processChunkGraph } from './graph.js';
import {
  basesWithPendingWork, finalizeReadyDocuments, getBase, pendingEmbeddingChunks, pendingGraphChunks, setBaseEmbedding, setChunkGraphStatus, storeEmbeddings,
} from './store.js';

const INTERVAL_MS = 3_000;
const RETENTION_INTERVAL_MS = 60 * 60 * 1000;
const EMBED_BATCH = 32;
const GRAPH_BATCH = 4;
const MAX_CONSECUTIVE_FAILURES = 5;

export interface IndexerReport {
  embedded: number;
  extracted: number;
  extractionErrors: number;
  documentsReady: number;
  errors: string[];
}

let cancelTick: (() => void) | null = null;
let cancelRetention: (() => void) | null = null;
let running = false;
let started = false;
const embedFailures = new Map<number, number>();
// Graph extraction retries per chunk: a rate-limited or momentarily dry
// fallback chain must not brand a chunk 'error' forever after one miss.
const graphAttempts = new Map<number, number>();
const MAX_GRAPH_ATTEMPTS = 3;

export async function runIndexerOnce(opts: { embedLimit?: number; graphLimit?: number } = {}): Promise<IndexerReport> {
  const report: IndexerReport = { embedded: 0, extracted: 0, extractionErrors: 0, documentsReady: 0, errors: [] };
  const policy = getPolicy();
  for (const baseId of basesWithPendingWork()) {
    const base = getBase(baseId);
    if (!base) continue;

    // 1. Embeddings.
    const pending = pendingEmbeddingChunks(base.id, opts.embedLimit ?? EMBED_BATCH);
    if (pending.length > 0) {
      try {
        const embedder = embedderFor(base.embedder || 'hash', base.embedding_model);
        const vectors = await embedder.embed(pending.map(c => c.text));
        storeEmbeddings(base.id, pending.map((c, i) => ({ chunkId: c.id, vector: vectors[i] })), embedder.model);
        if (base.embedding_dims === 0 || base.embedding_model !== embedder.model) {
          setBaseEmbedding(base.id, embedder.kind, embedder.model, vectors[0]?.length ?? 0);
        }
        report.embedded += pending.length;
        embedFailures.delete(base.id);
      } catch (err: any) {
        const message = String(err?.message ?? err).slice(0, 300);
        const count = (embedFailures.get(base.id) ?? 0) + 1;
        embedFailures.set(base.id, count);
        report.errors.push(`base ${base.slug}: embedding failed (${count}): ${message}`);
        console.warn(`[knowledge/indexer] embedding failed for base ${base.slug} (${count}/${MAX_CONSECUTIVE_FAILURES}): ${message}`);
        if (count >= MAX_CONSECUTIVE_FAILURES) {
          getDb().prepare("UPDATE knowledge_documents SET status = 'error', error = ?, updated_at_ms = ? WHERE base_id = ? AND status = 'indexing'")
            .run(`embedding failed: ${message}`, Date.now(), base.id);
          embedFailures.delete(base.id);
        }
        continue;
      }
    }

    // 2. Graph extraction (bounded: each chunk is a model call, and the
    //    policy's per-minute budget keeps free tiers usable for answers).
    if (policy.graph.extract) {
      let consecutiveErrors = 0;
      const perTick = Math.max(1, Math.min(GRAPH_BATCH, Math.floor(policy.graph.extractions_per_minute * INTERVAL_MS / 60_000)));
      for (const chunk of pendingGraphChunks(base.id, opts.graphLimit ?? perTick)) {
        const result = await processChunkGraph(base, chunk);
        if (result.status === 'done') {
          report.extracted++;
          consecutiveErrors = 0;
          graphAttempts.delete(chunk.id);
        } else {
          const attempts = (graphAttempts.get(chunk.id) ?? 0) + 1;
          if (attempts < MAX_GRAPH_ATTEMPTS) {
            // Back to the queue; the next tick retries after any cooldown.
            graphAttempts.set(chunk.id, attempts);
            setChunkGraphStatus(chunk.id, 'pending');
          } else {
            graphAttempts.delete(chunk.id);
            report.extractionErrors++;
          }
          report.errors.push(`base ${base.slug} chunk ${chunk.id} (attempt ${attempts}): ${result.error ?? 'extraction failed'}`);
          if (++consecutiveErrors >= 3) break;
        }
      }
    } else {
      getDb().prepare("UPDATE knowledge_chunks SET graph_status = 'skipped' WHERE base_id = ? AND graph_status = 'pending' AND embedding IS NOT NULL").run(base.id);
    }

    // 3. Documents whose chunks are all done.
    report.documentsReady += finalizeReadyDocuments(base.id, policy.graph.extract);
  }
  return report;
}

async function tick(): Promise<void> {
  if (running) return;
  running = true;
  try {
    const r = await runIndexerOnce();
    if (r.embedded || r.extracted || r.documentsReady) {
      console.log(`[knowledge/indexer] embedded ${r.embedded}, extracted ${r.extracted}, ready ${r.documentsReady}${r.extractionErrors ? `, extraction errors ${r.extractionErrors}` : ''}`);
    }
    if (r.errors.length) console.warn(`[knowledge/indexer] ${r.errors[0]}${r.errors.length > 1 ? ` (+${r.errors.length - 1} more)` : ''}`);
  } catch (err: any) {
    console.error(`[knowledge/indexer] tick failed: ${err?.message ?? err}`);
  } finally {
    running = false;
  }
}

export function startKnowledgeIndexer(scheduler: Scheduler): void {
  if (cancelTick) return;
  started = true;
  cancelTick = scheduler.every(INTERVAL_MS, tick, { name: 'knowledge-indexer' });
  cancelRetention = scheduler.every(RETENTION_INTERVAL_MS, () => {
    try {
      const swept = sweepRetention();
      if (swept.queries || swept.audit || swept.evalRuns) console.log(`[knowledge] retention sweep: ${swept.queries} provenance, ${swept.audit} audit, ${swept.evalRuns} eval runs`);
    } catch (err: any) {
      console.error(`[knowledge] retention sweep failed: ${err?.message ?? err}`);
    }
  }, { name: 'knowledge-retention' });
}

export function stopKnowledgeIndexer(): void {
  cancelTick?.();
  cancelRetention?.();
  cancelTick = null;
  cancelRetention = null;
  started = false;
}

/** Run a tick soon (after an ingest) instead of waiting for the interval. */
export function kickIndexer(): void {
  if (!started || running) return;
  setImmediate(() => { void tick(); });
}

export function indexerStatus(): { started: boolean; running: boolean } {
  return { started, running };
}
