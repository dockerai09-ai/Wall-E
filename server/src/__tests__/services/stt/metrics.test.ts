import { describe, it, expect } from 'vitest';
import {
  normalizeSwedish, normalizeSwedishText, parseSwedishNumber, wer, cer, align, containsTerm, leaksPii, scoreCase,
  aggregate, frequentMisses, percentile, maskPii,
} from '../../../services/stt/metrics.js';

describe('parseSwedishNumber', () => {
  it('handles units, tens, compounds, hundreds and thousands', () => {
    expect(parseSwedishNumber('fem')).toBe(5);
    expect(parseSwedishNumber('femtio')).toBe(50);
    expect(parseSwedishNumber('tjugofem')).toBe(25);
    expect(parseSwedishNumber('trettioåtta')).toBe(38);
    expect(parseSwedishNumber('hundra')).toBe(100);
    expect(parseSwedishNumber('femhundra')).toBe(500);
    expect(parseSwedishNumber('etthundrafemtio')).toBe(150);
    expect(parseSwedishNumber('tvåtusen')).toBe(2000);
    expect(parseSwedishNumber('tusenfemhundratjugo')).toBe(1520);
  });
  it('rejects ordinary words', () => {
    expect(parseSwedishNumber('bröstet')).toBeNull();
    expect(parseSwedishNumber('femton-')).toBeNull();
    expect(parseSwedishNumber('hundrade')).toBeNull();
  });
});

describe('normalizeSwedish', () => {
  it('maps number words, decimals and dose units so spoken and written forms compare equal', () => {
    expect(normalizeSwedishText('metoprolol femtio milligram')).toBe('metoprolol 50 mg');
    expect(normalizeSwedishText('Metoprolol 50 mg.')).toBe('metoprolol 50 mg');
    expect(normalizeSwedishText('50mg')).toBe('50 mg');
    expect(normalizeSwedishText('tolv komma fem')).toBe('12.5');
    expect(normalizeSwedishText('12,5')).toBe('12.5');
    expect(normalizeSwedishText('12 komma 5')).toBe('12.5');
    expect(normalizeSwedishText('hundra mikrogram')).toBe('100 mcg');
  });
  it('unifies spelling variants Whisper alternates between', () => {
    expect(normalizeSwedishText('sedan i går kväll')).toBe(normalizeSwedishText('sedan igår kväll'));
    expect(normalizeSwedishText('jag skall')).toBe('jag ska');
  });
  it('keeps digit tokens verbatim (leading zeros) and collapses spoken digit groups', () => {
    expect(normalizeSwedish('85 04 12 12 30')).toEqual(['8504121230']);
    expect(normalizeSwedish('850412-1230')).toEqual(['8504121230']);
    expect(normalizeSwedish('8504 12 12 30')).toEqual(['8504121230']);
    expect(normalizeSwedish('070 123 45 67')).toEqual(['0701234567']);
    expect(normalizeSwedish('två tabletter')).toEqual(['2', 'tabletter']);
  });
  it('drops redaction markers', () => {
    expect(normalizeSwedish('ringa [REDACTED:phone] nu')).toEqual(['ringa', 'nu']);
  });
});

describe('alignment and error rates', () => {
  it('counts substitutions, deletions and insertions', () => {
    expect(align(['a', 'b'], ['a', 'x']).map(o => o.op)).toEqual(['ok', 'sub']);
    expect(align(['a', 'b', 'c'], ['a', 'c']).map(o => o.op)).toEqual(['ok', 'del', 'ok']);
    expect(align(['a', 'c'], ['a', 'b', 'c']).map(o => o.op)).toEqual(['ok', 'ins', 'ok']);
  });
  it('WER is zero for equivalent spoken/written forms and counts real errors', () => {
    expect(wer('Jag tar metoprolol femtio milligram', 'Jag tar metoprolol 50 mg.')).toBe(0);
    expect(wer('Jag tar metoprolol', 'Jag tar metropolol')).toBeCloseTo(1 / 3);
    expect(wer('a b c d', 'a b')).toBe(0.5);
    expect(wer('', 'x')).toBe(1);
    expect(wer('', '')).toBe(0);
  });
  it('CER is character-level on the normalised text', () => {
    expect(cer('penicillin', 'penicilin')).toBeCloseTo(1 / 10);
  });
});

describe('terms and PII', () => {
  it('finds multi-word terms contiguously after normalisation', () => {
    const h = normalizeSwedish('det strålar ut i vänster arm, 50 mg');
    expect(containsTerm(h, 'vänster arm')).toBe(true);
    expect(containsTerm(h, 'femtio milligram')).toBe(true);
    expect(containsTerm(h, 'höger arm')).toBe(false);
  });
  it('detects leaked identifiers whatever the spacing', () => {
    expect(leaksPii('personnummer 85 04 12 12 30', '8504121230')).toBe(true);
    expect(leaksPii('personnummer 850412-1230', '8504121230')).toBe(true);
    expect(leaksPii('personnummer [REDACTED:personnummer]', '8504121230')).toBe(false);
  });
  it('masks protected values out of both sides so redaction is metric-neutral', () => {
    expect(maskPii(['pnr', '8504121230', 'ok'], ['8504121230'])).toEqual(['pnr', 'ok']);
    const redacted = scoreCase({ id: 'x', reference: 'Mitt personnummer är 85 04 12 12 30 och jag bor i Solna', hypothesis: 'Mitt personnummer är [REDACTED:personnummer] och jag bor i Solna', pii: ['8504121230'] });
    expect(redacted.wer).toBe(0);
    expect(redacted.piiLeaked).toBe(0);
    const leaked = scoreCase({ id: 'y', reference: 'Mitt personnummer är 85 04 12 12 30 och jag bor i Solna', hypothesis: 'Mitt personnummer är 8504 12 12 30 och jag bor i Solna', pii: ['8504121230'] });
    expect(leaked.wer).toBe(0);
    expect(leaked.piiLeaked).toBe(1);
  });
});

describe('scoreCase and aggregate', () => {
  it('scores a case with terms and reports missed words', () => {
    const s = scoreCase({ id: 'c', reference: 'Jag tar metoprolol och är allergisk mot penicillin', hypothesis: 'Jag tar metropolol och är allergisk mot penicillin', terms: ['metoprolol', 'penicillin'] });
    expect(s.substitutions).toBe(1);
    expect(s.termsFound).toBe(1);
    expect(s.termRecall).toBe(0.5);
    expect(s.missed).toEqual(['metoprolol']);
  });
  it('aggregates corpus-level rates, latencies and errors', () => {
    const a = scoreCase({ id: 'a', reference: 'ett två tre fyra', hypothesis: 'ett två tre fyra' });
    const b = scoreCase({ id: 'b', reference: 'a b c d', hypothesis: 'a b' });
    const m = aggregate([a, b], { errors: 1, latenciesMs: [100, 200, 300, 400], totalCases: 3 });
    expect(m.cases).toBe(3);
    expect(m.scored).toBe(2);
    expect(m.errors).toBe(1);
    expect(m.wer).toBe(0.25);
    expect(m.werMean).toBe(0.25);
    expect(m.latencyP50Ms).toBe(200);
    expect(m.latencyP95Ms).toBe(400);
    expect(percentile([], 50)).toBe(0);
  });
  it('frequentMisses skips stopwords, numbers and short tokens', () => {
    const s1 = scoreCase({ id: '1', reference: 'jag tar metoprolol femtio', hypothesis: 'jag tar metropolol fyrtio' });
    const s2 = scoreCase({ id: '2', reference: 'metoprolol och penicillin', hypothesis: 'metropolol och penicillin' });
    expect(frequentMisses([s1, s2], 2)).toEqual([{ term: 'metoprolol', count: 2 }]);
    expect(frequentMisses([s1, s2], 1).map(m => m.term)).toEqual(['metoprolol']);
  });
});
