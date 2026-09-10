// Document ingestion: policy checks, PII redaction, content hashing and
// dedupe, chunking, then a synchronous insert of document + chunks. Embedding
// and graph extraction happen in the background indexer so a large upload
// returns immediately with status 'indexing'.

import crypto from 'crypto';
import { chunkText } from './chunker.js';
import { KnowledgeError } from './config.js';
import { audit, getPolicy, isContentTypeAllowed, redactText } from './governance.js';
import { createDocumentWithChunks, findDocumentByHash, type BaseRow, type DocumentRow } from './store.js';
import { getDb } from '../../db/index.js';
import { kickIndexer } from './indexer.js';

export interface IngestInput {
  title?: string;
  text?: string;
  url?: string;
  source?: string;
  contentType?: string;
  metadata?: Record<string, unknown>;
  actor: string;
}

export interface IngestResult {
  document: DocumentRow;
  deduplicated: boolean;
  redactions: number;
  chunks: number;
}

const FETCH_TIMEOUT_MS = 30_000;
// NUL, zero-width space and BOM: invisible characters that break chunk hashes
// and FTS tokens without changing the text a human sees. Built from char
// codes so the source file itself never carries them.
const INVISIBLE_RE = new RegExp(`[${String.fromCharCode(0, 0x200b, 0xfeff)}]`, 'g');

function decodeEntities(s: string): string {
  return s.replace(/&(amp|lt|gt|quot|#39|nbsp);/g, (_, e: string) => ({ amp: '&', lt: '<', gt: '>', quot: '"', '#39': "'", nbsp: ' ' }[e] ?? ''));
}

/** Crude but dependency-free HTML -> text: drop script/style, keep block breaks. */
export function htmlToText(html: string): { text: string; title: string | null } {
  const title = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html)?.[1]?.trim() ?? null;
  const text = html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<\/(p|div|section|article|li|h[1-6]|tr|br|blockquote|pre)>/gi, '\n\n')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n[ \t]+/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  return { text: decodeEntities(text), title: title ? decodeEntities(title) : null };
}

async function fetchUrl(url: string, maxBytes: number): Promise<{ text: string; contentType: string; title: string | null }> {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new KnowledgeError('url is not valid');
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') throw new KnowledgeError('only http(s) URLs can be ingested');
  const res = await fetch(parsed, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS), headers: { Accept: 'text/html, text/plain, text/markdown, application/json;q=0.9, */*;q=0.1' } });
  if (!res.ok) throw new KnowledgeError(`fetching ${parsed.host} failed with ${res.status}`, 502, 'upstream_error');
  const declared = Number(res.headers.get('content-length') ?? 0);
  if (declared > maxBytes) throw new KnowledgeError(`document is larger than the policy limit (${maxBytes} bytes)`, 413);
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.byteLength > maxBytes) throw new KnowledgeError(`document is larger than the policy limit (${maxBytes} bytes)`, 413);
  const contentType = (res.headers.get('content-type') ?? 'text/plain').split(';')[0].trim().toLowerCase();
  const body = buf.toString('utf8');
  if (contentType === 'text/html' || contentType === 'application/xhtml+xml') {
    const { text, title } = htmlToText(body);
    return { text, contentType: 'text/html', title };
  }
  return { text: body, contentType, title: null };
}

export async function ingestDocument(base: BaseRow, input: IngestInput): Promise<IngestResult> {
  const policy = getPolicy();
  let text = input.text ?? '';
  let contentType = (input.contentType ?? 'text/plain').split(';')[0].trim().toLowerCase();
  let title = input.title?.trim() ?? '';
  let source = input.source?.trim() ?? '';

  if (input.url) {
    const fetched = await fetchUrl(input.url, policy.ingest.max_document_bytes);
    text = fetched.text;
    contentType = fetched.contentType;
    title = title || fetched.title || input.url;
    source = source || input.url;
  }

  if (!isContentTypeAllowed(contentType, policy)) {
    audit({ action: 'ingest.rejected', actor: input.actor, baseId: base.id, target: title || source, details: { reason: 'content_type', contentType } }, policy);
    throw new KnowledgeError(`content type '${contentType}' is not allowed by the governance policy`, 415);
  }
  const byteSize = Buffer.byteLength(text, 'utf8');
  if (byteSize > policy.ingest.max_document_bytes) {
    audit({ action: 'ingest.rejected', actor: input.actor, baseId: base.id, target: title || source, details: { reason: 'size', byteSize } }, policy);
    throw new KnowledgeError(`document is larger than the policy limit (${policy.ingest.max_document_bytes} bytes)`, 413);
  }
  if (contentType === 'text/html' && !input.url) {
    const converted = htmlToText(text);
    text = converted.text;
    title = title || converted.title || '';
  }
  text = text.replace(/\r\n?/g, '\n').replace(INVISIBLE_RE, '').trim();
  if (!text) throw new KnowledgeError('document has no text content');
  if (!title) title = source || text.split('\n')[0].slice(0, 80) || 'Untitled';

  const red = redactText(text, policy);
  const contentHash = crypto.createHash('sha256').update(red.text).digest('hex');

  const existing = findDocumentByHash(base.id, contentHash);
  if (existing) {
    if (existing.status !== 'deleted' && policy.ingest.dedupe_by_content_hash) {
      audit({ action: 'ingest.deduplicated', actor: input.actor, baseId: base.id, target: `document:${existing.id}`, details: { title } }, policy);
      return { document: existing, deduplicated: true, redactions: existing.redactions, chunks: existing.chunk_count };
    }
    // A tombstone (or dedupe disabled) with the same content: drop the old
    // row so the unique index accepts the new one.
    getDb().prepare('DELETE FROM knowledge_documents WHERE id = ?').run(existing.id);
  }

  const chunks = chunkText(red.text, { targetTokens: policy.ingest.chunk.target_tokens, overlapTokens: policy.ingest.chunk.overlap_tokens });
  const document = createDocumentWithChunks({
    baseId: base.id, title, source, contentType, contentHash, byteSize, metadata: input.metadata, redactions: red.total, chunks,
  });
  audit({
    action: 'ingest', actor: input.actor, baseId: base.id, target: `document:${document.id}`,
    details: { title, source, contentType, byteSize, chunks: chunks.length, redactions: red.redactions },
  }, policy);
  kickIndexer();
  return { document, deduplicated: false, redactions: red.total, chunks: chunks.length };
}
