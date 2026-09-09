// Embedding storage and similarity. Vectors are L2-normalised before they are
// written so cosine similarity is a plain dot product at query time. The
// search is brute force over one base's chunks with a per-base decoded cache;
// that is comfortably fast to ~100k chunks, which is far past the size of a
// single-user knowledge base. Swap in sqlite-vec here if that ever changes.

import { getDb } from '../../db/index.js';

export function normalize(vec: number[]): Float32Array {
  const out = new Float32Array(vec.length);
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
  const norm = Math.sqrt(sum) || 1;
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

export function encodeVector(vec: Float32Array): Buffer {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
}

export function decodeVector(blob: Buffer | Uint8Array): Float32Array {
  const buf = Buffer.isBuffer(blob) ? blob : Buffer.from(blob);
  // Copy: SQLite may hand back a view over a shared arena.
  const copy = Buffer.from(buf);
  return new Float32Array(copy.buffer, copy.byteOffset, Math.floor(copy.byteLength / 4));
}

export function dot(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

interface CachedVectors {
  version: number;
  rows: { id: number; documentId: number; vec: Float32Array }[];
}

const cache = new Map<number, CachedVectors>();
const versions = new Map<number, number>();

/** Call after any write to a base's chunks/embeddings. */
export function invalidateVectorCache(baseId: number): void {
  versions.set(baseId, (versions.get(baseId) ?? 0) + 1);
  cache.delete(baseId);
}

function loadVectors(baseId: number): CachedVectors['rows'] {
  const version = versions.get(baseId) ?? 0;
  const hit = cache.get(baseId);
  if (hit && hit.version === version) return hit.rows;
  const rows = getDb().prepare(
    'SELECT id, document_id, embedding FROM knowledge_chunks WHERE base_id = ? AND embedding IS NOT NULL',
  ).all(baseId) as { id: number; document_id: number; embedding: Buffer }[];
  const decoded = rows.map(r => ({ id: r.id, documentId: r.document_id, vec: decodeVector(r.embedding) }));
  cache.set(baseId, { version, rows: decoded });
  return decoded;
}

export interface VectorHit { chunkId: number; documentId: number; score: number }

export function vectorSearch(baseId: number, query: Float32Array, k: number): VectorHit[] {
  const rows = loadVectors(baseId);
  const hits: VectorHit[] = [];
  for (const row of rows) {
    if (row.vec.length !== query.length) continue; // stale dims after a re-embed
    hits.push({ chunkId: row.id, documentId: row.documentId, score: dot(row.vec, query) });
  }
  hits.sort((a, b) => b.score - a.score);
  return hits.slice(0, Math.max(0, k));
}
