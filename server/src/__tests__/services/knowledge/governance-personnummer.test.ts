import { describe, it, expect, afterEach } from 'vitest';
import { parsePolicy, redactText, setPolicyForTests } from '../../../services/knowledge/governance.js';

const policy = (detectors: string[]) => parsePolicy(`version: 1\nname: t\ningest:\n  pii_detectors: [${detectors.join(', ')}]\n`, '<test>');

describe('personnummer detector', () => {
  afterEach(() => setPolicyForTests(null));

  it('redacts valid ten- and twelve-digit numbers in every written form', () => {
    const p = policy(['personnummer', 'phone']);
    for (const s of ['850412-1230', '8504121230', '19850412-1230', '198504121230', '850412 1230', '850412+1230']) {
      const r = redactText(`pnr ${s} slut`, p);
      expect(r.text, s).toBe('pnr [REDACTED:personnummer] slut');
      expect(r.redactions).toEqual([{ detector: 'personnummer', count: 1 }]);
    }
  });

  it('leaves numbers that fail the check digit or the date alone, and does not steal phone numbers', () => {
    const only = policy(['personnummer']);
    expect(redactText('order 850412-1231', only).text).toBe('order 850412-1231');
    expect(redactText('ref 991399-1230', only).text).toBe('ref 991399-1230');
    expect(redactText('ring 0701234567', only).text).toBe('ring 0701234567');
    const both = policy(['personnummer', 'phone']);
    expect(redactText('ring 070-1234567', both).text).toBe('ring [REDACTED:phone]');
    expect(redactText('ring 0701234567 pnr 850412-1230', both).text).toBe('ring [REDACTED:phone] pnr [REDACTED:personnummer]');
  });

  it('is off unless the policy lists it', () => {
    expect(redactText('pnr 850412-1230', policy(['email'])).text).toBe('pnr 850412-1230');
  });
});
