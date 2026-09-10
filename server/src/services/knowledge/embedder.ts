// Embedder abstraction. Three implementations:
//   router — Wall-E's own embeddings engine (services/embeddings.ts), i.e. any
//            provider enabled on the dashboard's Embeddings page, with the
//            engine's within-family failover.
//   openai — any OpenAI-compatible /embeddings endpoint (OpenAI, Ollama, Jina,
//            Voyage, a local TEI server...). KB_EMBEDDING_BASE_URL + model.
//   hash   — deterministic feature hashing over word uni/bigrams and character
//            trigrams. No network, no keys, no model download. Quality is far
//            below a learned model; it exists so the module works out of the
//            box and so tests are hermetic. `auto` falls back to it with a
//            warning when the router has no usable embedding provider.
//
// The choice is made ONCE per knowledge base (knowledge_bases.embedder /
// embedding_model) so every chunk of a base lives in the same vector space.

import { runEmbeddings } from '../embeddings.js';
import { getKnowledgeConfig, KnowledgeError, type KnowledgeConfig } from './config.js';

export interface Embedder {
  readonly kind: 'router' | 'openai' | 'hash';
  /** Human-readable model identity, recorded on the base and in provenance. */
  readonly model: string;
  embed(texts: string[]): Promise<number[][]>;
}

export const HASH_DIMS = 256;

// FNV-1a 32-bit; the low bit of a second hash picks the sign so buckets
// cancel rather than pile up (sign hashing, Weinberger et al.).
function fnv1a(s: string, seed = 0x811c9dc5): number {
  let h = seed >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function hashFeatures(text: string): string[] {
  const words = text.toLowerCase().replace(/[^\p{L}\p{N}\s]+/gu, ' ').split(/\s+/).filter(Boolean);
  const feats: string[] = [];
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    feats.push(`w:${w}`);
    if (i + 1 < words.length) feats.push(`b:${w}_${words[i + 1]}`);
    if (w.length >= 5) {
      for (let j = 0; j + 3 <= w.length; j++) feats.push(`t:${w.slice(j, j + 3)}`);
    }
  }
  return feats;
}

export function hashEmbed(text: string, dims = HASH_DIMS): number[] {
  const vec = new Array<number>(dims).fill(0);
  const feats = hashFeatures(text);
  for (const f of feats) {
    const h = fnv1a(f);
    const idx = h % dims;
    const sign = (fnv1a(f, 0x9747b28c) & 1) === 0 ? 1 : -1;
    // Sub-linear term weighting keeps one repeated token from dominating.
    vec[idx] += sign;
  }
  for (let i = 0; i < dims; i++) vec[i] = Math.sign(vec[i]) * Math.log1p(Math.abs(vec[i]));
  return vec;
}

export class HashEmbedder implements Embedder {
  readonly kind = 'hash' as const;
  readonly model = `hash-${HASH_DIMS}`;
  async embed(texts: string[]): Promise<number[][]> {
    return texts.map(t => hashEmbed(t));
  }
}

const ROUTER_BATCH = 32;

export class RouterEmbedder implements Embedder {
  readonly kind = 'router' as const;
  model: string;
  private readonly requested: string;
  constructor(model: string | null) {
    this.requested = model ?? 'auto';
    this.model = `router:${this.requested}`;
  }
  async embed(texts: string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += ROUTER_BATCH) {
      const batch = texts.slice(i, i + ROUTER_BATCH);
      const res = await runEmbeddings(this.requested, batch);
      // Pin the identity to what actually served the first batch.
      this.model = `router:${res.family}`;
      out.push(...res.vectors);
    }
    return out;
  }
}

const OPENAI_BATCH = 64;
const OPENAI_TIMEOUT_MS = 60_000;

export class OpenAIEmbedder implements Embedder {
  readonly kind = 'openai' as const;
  readonly model: string;
  constructor(private readonly baseUrl: string, private readonly modelName: string, private readonly apiKey: string | null) {
    this.model = `openai:${modelName}`;
  }
  async embed(texts: string[]): Promise<number[][]> {
    const out: number[][] = [];
    const url = `${this.baseUrl.replace(/\/+$/, '')}/embeddings`;
    for (let i = 0; i < texts.length; i += OPENAI_BATCH) {
      const batch = texts.slice(i, i + OPENAI_BATCH);
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
      const r = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify({ model: this.modelName, input: batch }),
        signal: AbortSignal.timeout(OPENAI_TIMEOUT_MS),
      });
      if (!r.ok) throw new KnowledgeError(`embedding endpoint ${r.status}: ${(await r.text()).slice(0, 200)}`, 502, 'upstream_error');
      const j = (await r.json()) as { data?: { index?: number; embedding?: number[] }[] };
      const rows = (j.data ?? []).slice().sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
      if (rows.length !== batch.length || rows.some(row => !Array.isArray(row.embedding) || row.embedding.length === 0)) {
        throw new KnowledgeError('embedding endpoint returned malformed embeddings', 502, 'upstream_error');
      }
      out.push(...rows.map(row => row.embedding as number[]));
    }
    return out;
  }
}

/** Build the embedder for a base whose choice was already recorded. */
export function embedderFor(kind: string, model: string, cfg: KnowledgeConfig = getKnowledgeConfig()): Embedder {
  switch (kind) {
    case 'hash': return new HashEmbedder();
    case 'router': return new RouterEmbedder(model.startsWith('router:') ? model.slice('router:'.length) : cfg.embeddingModel);
    case 'openai': {
      if (!cfg.embeddingBaseUrl) throw new KnowledgeError('KB_EMBEDDING_BASE_URL is required for the openai embedder', 500, 'configuration_error');
      const name = model.startsWith('openai:') ? model.slice('openai:'.length) : (cfg.embeddingModel ?? 'text-embedding-3-small');
      return new OpenAIEmbedder(cfg.embeddingBaseUrl, name, cfg.embeddingApiKey);
    }
    default:
      throw new KnowledgeError(`unknown embedder '${kind}'`, 500, 'configuration_error');
  }
}

export interface EmbedderChoice { embedder: Embedder; warning: string | null }

/**
 * Decide the embedder for a NEW base. `auto` probes the router once (one tiny
 * embeddings call) so a base never silently starts in the weak hash space
 * when a real provider is available — and never gets stuck when none is.
 */
export async function chooseEmbedder(cfg: KnowledgeConfig = getKnowledgeConfig()): Promise<EmbedderChoice> {
  switch (cfg.embedder) {
    case 'hash': return { embedder: new HashEmbedder(), warning: null };
    case 'router': return { embedder: new RouterEmbedder(cfg.embeddingModel), warning: null };
    case 'openai': return { embedder: embedderFor('openai', '', cfg), warning: null };
    case 'auto':
    default: {
      const router = new RouterEmbedder(cfg.embeddingModel);
      try {
        const probe = await router.embed(['knowledge base embedder probe']);
        if (probe.length === 1 && probe[0].length > 0) return { embedder: router, warning: null };
      } catch (err: any) {
        const reason = String(err?.message ?? err).slice(0, 160);
        return {
          embedder: new HashEmbedder(),
          warning: `No usable embedding provider (${reason}); using the local hash embedder. Enable a provider on the Embeddings page or set KB_EMBEDDER for better retrieval.`,
        };
      }
      return { embedder: new HashEmbedder(), warning: 'Embedding probe returned nothing; using the local hash embedder.' };
    }
  }
}
