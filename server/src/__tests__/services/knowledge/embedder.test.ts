import { describe, it, expect, afterEach, vi } from 'vitest';
import { HashEmbedder, OpenAIEmbedder, hashEmbed, HASH_DIMS, chooseEmbedder, embedderFor } from '../../../services/knowledge/embedder.js';
import { normalize, dot, encodeVector, decodeVector } from '../../../services/knowledge/vector.js';
import type { KnowledgeConfig } from '../../../services/knowledge/config.js';

const realFetch = globalThis.fetch;

const cfg = (over: Partial<KnowledgeConfig>): KnowledgeConfig => ({
  enabled: true, embedder: 'hash', embeddingModel: null, embeddingBaseUrl: null, embeddingApiKey: null,
  ontologyPath: '', governancePath: '', neo4j: null, ...over,
});

describe('hash embedder', () => {
  it('is deterministic and dimension-stable', async () => {
    const e = new HashEmbedder();
    const [a, b] = await e.embed(['Neo4j stores graphs', 'Neo4j stores graphs']);
    expect(a).toEqual(b);
    expect(a).toHaveLength(HASH_DIMS);
    expect(e.model).toBe(`hash-${HASH_DIMS}`);
  });

  it('places related texts closer than unrelated ones', () => {
    const q = normalize(hashEmbed('how do I rotate the encryption key'));
    const near = normalize(hashEmbed('Rotating the encryption key: run rotate-encryption-key with --new-key'));
    const far = normalize(hashEmbed('The Docker image builds natively on arm64 runners'));
    expect(dot(q, near)).toBeGreaterThan(dot(q, far));
  });

  it('round-trips through the BLOB encoding', () => {
    const v = normalize(hashEmbed('blob round trip'));
    const back = decodeVector(encodeVector(v));
    expect(back.length).toBe(v.length);
    expect(Math.abs(dot(v, back) - 1)).toBeLessThan(1e-6);
  });
});

describe('openai-compatible embedder', () => {
  afterEach(() => { globalThis.fetch = realFetch; vi.restoreAllMocks(); });

  it('posts batches and orders vectors by index', async () => {
    const calls: { url: string; body: any; auth: string | undefined }[] = [];
    globalThis.fetch = vi.fn(async (url: any, init: any) => {
      const body = JSON.parse(init.body);
      calls.push({ url: String(url), body, auth: init.headers.Authorization });
      const data = body.input.map((_: string, i: number) => ({ index: i, embedding: [i + 1, 0, 0] })).reverse();
      return new Response(JSON.stringify({ data }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }) as any;
    const e = new OpenAIEmbedder('http://127.0.0.1:11434/v1/', 'nomic-embed-text', 'secret');
    const out = await e.embed(['a', 'b']);
    expect(calls[0].url).toBe('http://127.0.0.1:11434/v1/embeddings');
    expect(calls[0].auth).toBe('Bearer secret');
    expect(calls[0].body.model).toBe('nomic-embed-text');
    expect(out).toEqual([[1, 0, 0], [2, 0, 0]]);
    expect(e.model).toBe('openai:nomic-embed-text');
  });

  it('rejects malformed responses', async () => {
    globalThis.fetch = vi.fn(async () => new Response(JSON.stringify({ data: [{ index: 0, embedding: [] }] }), { status: 200 })) as any;
    const e = new OpenAIEmbedder('http://x', 'm', null);
    await expect(e.embed(['a'])).rejects.toThrow(/malformed/);
  });

  it('surfaces upstream HTTP errors with status 502', async () => {
    globalThis.fetch = vi.fn(async () => new Response('nope', { status: 500 })) as any;
    const e = new OpenAIEmbedder('http://x', 'm', null);
    await expect(e.embed(['a'])).rejects.toMatchObject({ status: 502 });
  });
});

describe('chooseEmbedder / embedderFor', () => {
  it('honours an explicit hash choice without probing', async () => {
    const c = await chooseEmbedder(cfg({ embedder: 'hash' }));
    expect(c.embedder.kind).toBe('hash');
    expect(c.warning).toBeNull();
  });

  it('requires a base URL for the openai embedder', () => {
    expect(() => embedderFor('openai', 'openai:m', cfg({ embedder: 'openai' }))).toThrow(/KB_EMBEDDING_BASE_URL/);
    const e = embedderFor('openai', 'openai:m', cfg({ embedder: 'openai', embeddingBaseUrl: 'http://x' }));
    expect(e.model).toBe('openai:m');
  });

  it('rejects unknown kinds', () => {
    expect(() => embedderFor('bogus', '', cfg({}))).toThrow(/unknown embedder/);
  });
});

describe('llm helpers', () => {
  it('parses the router\'s announced reset time', async () => {
    const { parseResetMs, extractJsonObject } = await import('../../../services/knowledge/llm.js');
    expect(parseResetMs('All models exhausted: 300 routes checked. Soonest reset ~18s.')).toBe(18000);
    expect(parseResetMs('no available key for model')).toBeNull();
    expect(parseResetMs('All models exhausted: 300 routes checked (3 rate-limited or on cooldown, 2 failed earlier this request).')).toBe(15000);
    expect(extractJsonObject('Sure! ```json\n{"a": [1, 2,], }\n```')).toEqual({ a: [1, 2] });
    expect(extractJsonObject('no json here')).toBeNull();
  });
});
