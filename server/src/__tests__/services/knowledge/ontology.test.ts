import { describe, it, expect } from 'vitest';
import { parseOntology, isValidRelation, isValidClass, ontologyPromptBlock, loadOntology, setOntologyForTests, getOntology } from '../../../services/knowledge/ontology.js';
import { validateExtraction, extractionMessages } from '../../../services/knowledge/graph.js';
import { parsePolicy } from '../../../services/knowledge/governance.js';

const yaml = `
version: 2
name: test
classes:
  - name: Person
  - name: Organization
  - name: Technology
relations:
  - name: WORKS_FOR
    from: [Person]
    to: [Organization]
  - name: USES
    from: [Person, Organization]
    to: [Technology]
  - name: RELATED_TO
    from: [any]
    to: [any]
extraction:
  min_confidence: 0.6
`;

describe('ontology', () => {
  it('parses, indexes and hashes', () => {
    const o = parseOntology(yaml);
    expect(o.name).toBe('test');
    expect(o.hash).toMatch(/^[0-9a-f]{12}$/);
    expect(isValidClass(o, 'Person')).toBe(true);
    expect(isValidClass(o, 'Robot')).toBe(false);
    expect(isValidRelation(o, 'WORKS_FOR', 'Person', 'Organization')).toBe(true);
    expect(isValidRelation(o, 'WORKS_FOR', 'Organization', 'Person')).toBe(false);
    expect(isValidRelation(o, 'RELATED_TO', 'Technology', 'Person')).toBe(true);
    expect(isValidRelation(o, 'NOPE', 'Person', 'Person')).toBe(false);
    expect(o.extraction.min_confidence).toBe(0.6);
    expect(o.extraction.max_entities_per_chunk).toBe(25);
  });

  it('rejects relations that reference unknown classes and bad identifiers', () => {
    expect(() => parseOntology(yaml.replace('to: [Organization]', 'to: [Company]'))).toThrow(/unknown class 'Company'/);
    expect(() => parseOntology(yaml.replace('name: WORKS_FOR', 'name: worksFor'))).toThrow(/UPPER_SNAKE/);
    expect(() => parseOntology('version: 1\nname: x\nclasses: []\nrelations: []\n')).toThrow(/schema error/);
    expect(() => parseOntology('::: not yaml')).toThrow(/YAML is invalid|schema error/);
  });

  it('renders a prompt block naming every class and relation', () => {
    const block = ontologyPromptBlock(parseOntology(yaml));
    expect(block).toContain('- Person');
    expect(block).toContain('WORKS_FOR (Person -> Organization)');
    expect(extractionMessages('text', parseOntology(yaml))[0].content).toContain('RELATED_TO');
  });

  it('loads the shipped ontology and falls back to the built-in one when the file is missing', () => {
    setOntologyForTests(null);
    const shipped = getOntology();
    expect(shipped.name).toBe('wall-e-default');
    expect(shipped.classNames.has('Technology')).toBe(true);
    const fallback = loadOntology('/nonexistent/ontology.yaml');
    expect(fallback.name).toBe('builtin-minimal');
    setOntologyForTests(null);
  });
});

describe('extraction validation', () => {
  const o = parseOntology(yaml);
  const policy = parsePolicy('version: 1\nname: t\ngraph:\n  min_confidence: 0.5\n');

  it('keeps only ontology-conformant, confident, deduplicated items', () => {
    const v = validateExtraction({
      entities: [
        { name: 'Ada', class: 'Person', confidence: 0.9 },
        { name: 'ada', class: 'Person', confidence: 0.8 },
        { name: 'Acme', class: 'Organization', confidence: 0.9 },
        { name: 'Rust', class: 'Technology', confidence: 0.3 },
        { name: 'Mars', class: 'Planet', confidence: 0.9 },
      ],
      relations: [
        { from: 'Ada', to: 'Acme', type: 'WORKS_FOR', confidence: 0.9 },
        { from: 'Acme', to: 'Ada', type: 'WORKS_FOR', confidence: 0.9 },
        { from: 'Ada', to: 'Rust', type: 'USES', confidence: 0.9 },
        { from: 'Ada', to: 'Acme', type: 'RELATED_TO', confidence: 0.4 },
        { from: 'Ada', to: 'Ada', type: 'RELATED_TO', confidence: 0.9 },
      ],
    }, o, policy);
    expect(v.entities.map(e => e.name)).toEqual(['Ada', 'Acme']);
    expect(v.relations).toEqual([{ from: 'Ada', to: 'Acme', type: 'WORKS_FOR', confidence: 0.9 }]);
    expect(v.dropped).toEqual({ entities: 2, relations: 4 });
  });

  it('tolerates garbage', () => {
    expect(validateExtraction(null, o, policy).entities).toEqual([]);
    expect(validateExtraction({ entities: 'x' }, o, policy).relations).toEqual([]);
  });
});
