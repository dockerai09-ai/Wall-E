// Ontology: the schema the knowledge graph is allowed to use. Loaded from
// knowledge/ontology.yaml (KB_ONTOLOGY_PATH), validated with zod, cached in
// memory, and hashed so provenance records can name the exact schema an
// extraction ran under. Neo4j gets the same schema as (:OntologyClass) and
// (:OntologyRelation) nodes plus constraints (see neo4j.ts).

import fs from 'fs';
import crypto from 'crypto';
import { parse as parseYaml } from 'yaml';
import { z } from 'zod';
import { getKnowledgeConfig, KnowledgeError } from './config.js';

const CLASS_NAME = /^[A-Z][A-Za-z0-9]*$/;
const RELATION_NAME = /^[A-Z][A-Z0-9_]*$/;

const classSchema = z.object({
  name: z.string().regex(CLASS_NAME, 'class names are PascalCase identifiers'),
  description: z.string().optional(),
  properties: z.array(z.string()).default([]),
});

const relationSchema = z.object({
  name: z.string().regex(RELATION_NAME, 'relation names are UPPER_SNAKE identifiers'),
  description: z.string().optional(),
  from: z.array(z.string()).min(1),
  to: z.array(z.string()).min(1),
});

const extractionSchema = z.object({
  max_entities_per_chunk: z.number().int().positive().default(25),
  max_relations_per_chunk: z.number().int().positive().default(40),
  min_confidence: z.number().min(0).max(1).default(0.5),
}).default({});

const ontologySchema = z.object({
  version: z.number().int().nonnegative(),
  name: z.string().min(1),
  description: z.string().optional(),
  classes: z.array(classSchema).min(1),
  relations: z.array(relationSchema).min(1),
  system: z.unknown().optional(),
  extraction: extractionSchema,
});

export type OntologyClass = z.infer<typeof classSchema>;
export type OntologyRelation = z.infer<typeof relationSchema>;
export type OntologyDoc = z.infer<typeof ontologySchema>;

export interface Ontology extends OntologyDoc {
  /** sha256 of the canonical JSON, first 12 hex chars. */
  hash: string;
  path: string;
  loadedAtMs: number;
  classNames: Set<string>;
  relationsByName: Map<string, OntologyRelation>;
}

// Used when no ontology file is reachable (e.g. a container image built
// without the knowledge/ directory). Small on purpose: the YAML is the real one.
const BUILTIN_ONTOLOGY: OntologyDoc = {
  version: 0,
  name: 'builtin-minimal',
  classes: [
    { name: 'Person', properties: ['name'] },
    { name: 'Organization', properties: ['name'] },
    { name: 'Product', properties: ['name'] },
    { name: 'Technology', properties: ['name'] },
    { name: 'Concept', properties: ['name'] },
  ],
  relations: [
    { name: 'USES', from: ['Product', 'Organization', 'Person'], to: ['Technology', 'Product'] },
    { name: 'PART_OF', from: ['any'], to: ['any'] },
    { name: 'RELATED_TO', from: ['any'], to: ['any'] },
  ],
  extraction: { max_entities_per_chunk: 25, max_relations_per_chunk: 40, min_confidence: 0.5 },
};

let cached: Ontology | null = null;

function finish(doc: OntologyDoc, path: string): Ontology {
  const classNames = new Set(doc.classes.map(c => c.name));
  for (const rel of doc.relations) {
    for (const side of [...rel.from, ...rel.to]) {
      if (side !== 'any' && !classNames.has(side)) {
        throw new KnowledgeError(`ontology relation ${rel.name} references unknown class '${side}'`, 500, 'configuration_error');
      }
    }
  }
  const canonical = JSON.stringify({ classes: doc.classes, relations: doc.relations, extraction: doc.extraction });
  return {
    ...doc,
    hash: crypto.createHash('sha256').update(canonical).digest('hex').slice(0, 12),
    path,
    loadedAtMs: Date.now(),
    classNames,
    relationsByName: new Map(doc.relations.map(r => [r.name, r])),
  };
}

export function parseOntology(yamlText: string, path = '<inline>'): Ontology {
  let raw: unknown;
  try {
    raw = parseYaml(yamlText);
  } catch (err: any) {
    throw new KnowledgeError(`ontology YAML is invalid: ${err?.message ?? err}`, 500, 'configuration_error');
  }
  const parsed = ontologySchema.safeParse(raw);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    throw new KnowledgeError(`ontology schema error at ${issue.path.join('.') || '<root>'}: ${issue.message}`, 500, 'configuration_error');
  }
  return finish(parsed.data, path);
}

export function loadOntology(path = getKnowledgeConfig().ontologyPath): Ontology {
  if (!fs.existsSync(path)) {
    console.warn(`[knowledge] ontology file not found at ${path}; using the built-in minimal ontology`);
    return finish(BUILTIN_ONTOLOGY, '<builtin>');
  }
  return parseOntology(fs.readFileSync(path, 'utf8'), path);
}

export function getOntology(): Ontology {
  if (!cached) cached = loadOntology();
  return cached;
}

export function reloadOntology(): Ontology {
  cached = loadOntology();
  return cached;
}

/** Tests and the reload endpoint can swap the active ontology explicitly. */
export function setOntologyForTests(o: Ontology | null): void {
  cached = o;
}

export function isValidClass(o: Ontology, cls: string): boolean {
  return o.classNames.has(cls);
}

export function isValidRelation(o: Ontology, type: string, fromClass: string, toClass: string): boolean {
  const rel = o.relationsByName.get(type);
  if (!rel) return false;
  const okFrom = rel.from.includes('any') || rel.from.includes(fromClass);
  const okTo = rel.to.includes('any') || rel.to.includes(toClass);
  return okFrom && okTo;
}

/** Compact schema description for the extraction prompt. */
export function ontologyPromptBlock(o: Ontology): string {
  const classes = o.classes.map(c => `- ${c.name}${c.description ? `: ${c.description.trim()}` : ''}`).join('\n');
  const relations = o.relations.map(r => `- ${r.name} (${r.from.join('|')} -> ${r.to.join('|')})`).join('\n');
  return `Entity classes:\n${classes}\n\nRelation types (from -> to):\n${relations}`;
}

export function ontologySummary(o: Ontology) {
  return {
    name: o.name,
    version: o.version,
    hash: o.hash,
    path: o.path,
    loadedAt: new Date(o.loadedAtMs).toISOString(),
    classes: o.classes,
    relations: o.relations,
    extraction: o.extraction,
  };
}
