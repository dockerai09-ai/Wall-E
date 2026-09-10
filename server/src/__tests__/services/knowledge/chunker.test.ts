import { describe, it, expect } from 'vitest';
import { chunkText, estimateTokens } from '../../../services/knowledge/chunker.js';

describe('knowledge chunker', () => {
  it('returns nothing for empty input', () => {
    expect(chunkText('')).toEqual([]);
    expect(chunkText('   \n\n  ')).toEqual([]);
  });

  it('keeps a short document as one chunk with correct offsets', () => {
    const text = 'Wall-E routes requests.\n\nIt has a dashboard.';
    const chunks = chunkText(text, { targetTokens: 400 });
    expect(chunks).toHaveLength(1);
    expect(chunks[0].ordinal).toBe(0);
    expect(chunks[0].charStart).toBe(0);
    expect(chunks[0].charEnd).toBe(text.length);
    expect(chunks[0].text).toBe(text);
    expect(chunks[0].tokenCount).toBe(estimateTokens(text));
  });

  it('splits on paragraphs, respects the target and overlaps neighbours', () => {
    const paras = Array.from({ length: 12 }, (_, i) => `Paragraph ${i} ${'lorem ipsum dolor sit amet '.repeat(12)}`.trim());
    const text = paras.join('\n\n');
    const chunks = chunkText(text, { targetTokens: 120, overlapTokens: 20 });
    expect(chunks.length).toBeGreaterThan(3);
    for (const c of chunks) expect(c.tokenCount).toBeLessThanOrEqual(120 + 20 + 10);
    // Offsets are monotonic and point at real content.
    for (let i = 1; i < chunks.length; i++) {
      expect(chunks[i].charStart).toBeGreaterThan(chunks[i - 1].charStart);
      expect(text.slice(chunks[i].charStart, chunks[i].charEnd).startsWith('Paragraph')).toBe(true);
    }
    // Overlap: the second chunk starts with the (word-aligned) tail of the
    // first chunk's body, then its own body.
    const firstBody = text.slice(chunks[0].charStart, chunks[0].charEnd);
    const secondBody = text.slice(chunks[1].charStart, chunks[1].charEnd);
    expect(chunks[1].text.endsWith(secondBody)).toBe(true);
    const overlap = chunks[1].text.slice(0, chunks[1].text.length - secondBody.length).trim();
    expect(overlap.length).toBeGreaterThan(0);
    expect(overlap.length).toBeLessThanOrEqual(20 * 4);
    expect(firstBody.endsWith(overlap)).toBe(true);
    expect(chunkText(text, { targetTokens: 120, overlapTokens: 0 })[1].text).toBe(secondBody);
  });

  it('hard-splits a single oversized sentence', () => {
    const text = 'x'.repeat(5000);
    const chunks = chunkText(text, { targetTokens: 100, overlapTokens: 0 });
    expect(chunks.length).toBeGreaterThanOrEqual(12);
    expect(chunks.map(c => c.text).join('')).toBe(text);
  });

  it('normalises CRLF line endings', () => {
    const chunks = chunkText('one\r\n\r\ntwo');
    expect(chunks[0].text).toBe('one\n\ntwo');
  });
});
