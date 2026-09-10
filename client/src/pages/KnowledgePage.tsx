import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { BookOpen, Database, FlaskConical, Network, Plus, RefreshCw, Send, Trash2 } from 'lucide-react'
import { PageHeader } from '@/components/page-header'
import { EmptyState } from '@/components/empty-state'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { apiFetch } from '@/lib/api'
import { toast } from '@/lib/toast'
import { cn } from '@/lib/utils'
import { useI18n } from '@/i18n'

// Wire shapes of /api/knowledge (server/src/routes/knowledge.ts). Kept local
// like the other pages do; shared/types.ts carries the canonical copies.
interface BaseStats { documents: number; documentsReady: number; chunks: number; chunksEmbedded: number; chunksGraphPending: number; entities: number; relations: number; queries: number }
interface KnowledgeBase { id: number; slug: string; name: string; description: string; embedder: string; embeddingModel: string; stats?: BaseStats }
interface KnowledgeDocument { id: number; title: string; source: string; status: 'indexing' | 'ready' | 'error' | 'deleted'; error: string | null; chunkCount: number; redactions: number; createdAt: string | null }
interface Citation { n: number; chunkId: number; documentId: number; title: string; source: string; ordinal: number }
interface SourceHit { chunkId: number; documentTitle: string; source: string; ordinal: number; text: string; score: number; via: string[] }
interface Answer {
  id: string
  answer: string
  status: 'ok' | 'refused'
  citations: Citation[]
  sources: SourceHit[]
  graph: { engine: string; nodes: { id: number; name: string; class: string }[]; edges: unknown[] }
  model: { platform: string; modelId: string } | null
  latencyMs: number
  governance: { citationsMissing: boolean; redactions: number; policyVersion: string }
}
interface Status {
  enabled: boolean
  embedder: { configured: string; model: string | null }
  ontology: { name: string; version: number; hash: string }
  governance: { version: string }
  neo4j: { configured: boolean; connected: boolean; error: string | null }
  indexer: { started: boolean; running: boolean }
  fts: boolean
}
interface EvalRun { id: string; dataset: string; status: string; cases: number; metrics: Record<string, number | null | { catchAllRatio: number }>; startedAt: string | null }

const STATUS_CLASS: Record<KnowledgeDocument['status'], string> = {
  ready: 'bg-emerald-600/15 text-emerald-700 dark:text-emerald-400',
  indexing: 'bg-sky-600/15 text-sky-700 dark:text-sky-400',
  error: 'bg-destructive/10 text-destructive',
  deleted: 'bg-muted text-muted-foreground',
}

function pct(v: unknown): string {
  return typeof v === 'number' ? `${(v * 100).toFixed(0)}%` : '–'
}

export function KnowledgePage() {
  const { t } = useI18n()
  const queryClient = useQueryClient()
  const [selected, setSelected] = useState<string | null>(null)
  const [newBaseName, setNewBaseName] = useState('')
  const [docTitle, setDocTitle] = useState('')
  const [docText, setDocText] = useState('')
  const [docUrl, setDocUrl] = useState('')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [showSources, setShowSources] = useState(false)
  const [datasetChoice, setDatasetChoice] = useState<string>('')
  const [judge, setJudge] = useState(false)

  const status = useQuery<Status>({ queryKey: ['knowledge', 'status'], queryFn: () => apiFetch('/api/knowledge/status') })
  const bases = useQuery<{ bases: KnowledgeBase[] }>({ queryKey: ['knowledge', 'bases'], queryFn: () => apiFetch('/api/knowledge/bases') })
  const baseList = useMemo(() => bases.data?.bases ?? [], [bases.data])
  const base = baseList.find(b => b.slug === selected) ?? baseList[0] ?? null
  const slug = base?.slug ?? null

  const hasPendingWork = !!base?.stats && (base.stats.chunksEmbedded < base.stats.chunks || base.stats.chunksGraphPending > 0)
  const docs = useQuery<{ documents: KnowledgeDocument[] }>({
    queryKey: ['knowledge', 'docs', slug],
    queryFn: () => apiFetch(`/api/knowledge/bases/${slug}/documents`),
    enabled: !!slug,
    refetchInterval: hasPendingWork ? 3000 : false,
  })
  useEffect(() => {
    if (!hasPendingWork) return
    const id = setInterval(() => queryClient.invalidateQueries({ queryKey: ['knowledge', 'bases'] }), 3000)
    return () => clearInterval(id)
  }, [hasPendingWork, queryClient])

  const datasets = useQuery<{ datasets: string[] }>({ queryKey: ['knowledge', 'datasets'], queryFn: () => apiFetch('/api/knowledge/evals/datasets') })
  const evalRuns = useQuery<{ runs: EvalRun[] }>({
    queryKey: ['knowledge', 'evals', slug],
    queryFn: () => apiFetch(`/api/knowledge/evals?base=${encodeURIComponent(slug ?? '')}&limit=10`),
    enabled: !!slug,
  })

  const invalidateBase = () => {
    queryClient.invalidateQueries({ queryKey: ['knowledge', 'bases'] })
    queryClient.invalidateQueries({ queryKey: ['knowledge', 'docs', slug] })
  }

  const createBase = useMutation({
    mutationFn: (name: string) => apiFetch<{ base: KnowledgeBase; warning: string | null }>('/api/knowledge/bases', { method: 'POST', body: JSON.stringify({ name }) }),
    onSuccess: (data) => {
      setNewBaseName('')
      setSelected(data.base.slug)
      queryClient.invalidateQueries({ queryKey: ['knowledge', 'bases'] })
      if (data.warning) toast.info(data.warning)
      else toast.success(t('knowledge.baseCreated'))
    },
  })

  const deleteBase = useMutation({
    mutationFn: (s: string) => apiFetch(`/api/knowledge/bases/${s}`, { method: 'DELETE' }),
    onSuccess: () => { setSelected(null); setAnswer(null); queryClient.invalidateQueries({ queryKey: ['knowledge'] }) },
  })

  const addDocument = useMutation({
    mutationFn: (body: { title?: string; text?: string; url?: string }) =>
      apiFetch<{ deduplicated: boolean; chunks: number; redactions: number }>(`/api/knowledge/bases/${slug}/documents`, { method: 'POST', body: JSON.stringify(body) }),
    onSuccess: (r) => {
      setDocTitle(''); setDocText(''); setDocUrl('')
      invalidateBase()
      toast.success(r.deduplicated ? t('knowledge.docDuplicate') : t('knowledge.docAdded', { chunks: r.chunks, redactions: r.redactions }))
    },
  })

  const deleteDocument = useMutation({
    mutationFn: (id: number) => apiFetch(`/api/knowledge/documents/${id}`, { method: 'DELETE' }),
    onSuccess: invalidateBase,
  })

  const indexNow = useMutation({
    mutationFn: () => apiFetch<{ embedded: number; extracted: number; documentsReady: number; errors: string[] }>(`/api/knowledge/bases/${slug}/index-now`, { method: 'POST' }),
    onSuccess: (r) => {
      invalidateBase()
      if (r.errors.length) toast.error(r.errors[0])
      else toast.success(t('knowledge.indexed', { embedded: r.embedded, extracted: r.extracted }))
    },
  })

  const reindex = useMutation({
    mutationFn: () => apiFetch(`/api/knowledge/bases/${slug}/reindex`, { method: 'POST' }),
    onSuccess: () => { invalidateBase(); toast.success(t('knowledge.reindexing')) },
  })

  const ask = useMutation({
    mutationFn: (q: string) => apiFetch<Answer>(`/api/knowledge/bases/${slug}/query`, { method: 'POST', body: JSON.stringify({ question: q }) }),
    onSuccess: (a) => { setAnswer(a); setShowSources(false); queryClient.invalidateQueries({ queryKey: ['knowledge', 'bases'] }) },
  })

  const runEval = useMutation({
    mutationFn: () => apiFetch<{ run: EvalRun }>('/api/knowledge/evals/run', { method: 'POST', body: JSON.stringify({ base: slug, dataset, judge }) }),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['knowledge', 'evals', slug] }); toast.success(t('knowledge.evalDone')) },
  })

  // Derived, not synced through an effect: the first dataset is the default
  // until the user picks another one.
  const dataset = datasetChoice || datasets.data?.datasets[0] || ''

  const st = status.data

  return (
    <div>
      <PageHeader
        title={t('knowledge.title')}
        description={t('knowledge.description')}
        actions={st && (
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline"><Database className="mr-1 size-3" />{t('knowledge.embedder')}: {st.embedder.configured}</Badge>
            <Badge variant="outline" className={cn(st.neo4j.connected && 'text-emerald-700 dark:text-emerald-400')}>
              <Network className="mr-1 size-3" />Neo4j: {st.neo4j.connected ? t('knowledge.connected') : st.neo4j.configured ? t('knowledge.unreachable') : t('knowledge.notConfigured')}
            </Badge>
            <Badge variant="outline">FTS5: {st.fts ? 'on' : 'off'}</Badge>
            <Badge variant="outline">{st.ontology.name} v{st.ontology.version}</Badge>
          </div>
        )}
      />

      {bases.isLoading ? (
        <Skeleton className="h-40 w-full" />
      ) : baseList.length === 0 ? (
        <EmptyState
          icon={BookOpen}
          title={t('knowledge.noBases')}
          description={t('knowledge.noBasesHint')}
          action={(
            <form className="flex items-center gap-2" onSubmit={e => { e.preventDefault(); if (newBaseName.trim()) createBase.mutate(newBaseName.trim()) }}>
              <Input value={newBaseName} onChange={e => setNewBaseName(e.target.value)} placeholder={t('knowledge.baseName')} className="w-56" />
              <Button type="submit" disabled={!newBaseName.trim() || createBase.isPending}><Plus className="size-4" />{t('knowledge.create')}</Button>
            </form>
          )}
        />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
          {/* Bases */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">{t('knowledge.bases')}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-1">
                  {baseList.map(b => (
                    <button
                      key={b.slug}
                      type="button"
                      onClick={() => { setSelected(b.slug); setAnswer(null) }}
                      className={cn('w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-muted', b.slug === slug && 'bg-muted font-medium')}
                    >
                      <div className="truncate">{b.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {b.stats?.documents ?? 0} docs · {b.stats?.chunks ?? 0} {t('knowledge.chunks')} · {b.stats?.entities ?? 0} {t('knowledge.entities')}
                      </div>
                    </button>
                  ))}
                </div>
                <form className="flex items-center gap-2" onSubmit={e => { e.preventDefault(); if (newBaseName.trim()) createBase.mutate(newBaseName.trim()) }}>
                  <Input value={newBaseName} onChange={e => setNewBaseName(e.target.value)} placeholder={t('knowledge.baseName')} />
                  <Button type="submit" size="icon" variant="outline" disabled={!newBaseName.trim() || createBase.isPending} aria-label={t('knowledge.create')}><Plus className="size-4" /></Button>
                </form>
              </CardContent>
            </Card>

            {base && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">{base.name}</CardTitle>
                  <CardDescription className="text-xs">{base.embeddingModel || base.embedder}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-xs text-muted-foreground">
                  <div>{base.stats?.documentsReady ?? 0}/{base.stats?.documents ?? 0} {t('knowledge.ready')} · {base.stats?.chunksEmbedded ?? 0}/{base.stats?.chunks ?? 0} embedded</div>
                  <div>{base.stats?.entities ?? 0} {t('knowledge.entities')} · {base.stats?.relations ?? 0} {t('knowledge.relations')} · {base.stats?.queries ?? 0} queries</div>
                  <div className="flex flex-wrap gap-2 pt-2">
                    <Button size="sm" variant="outline" onClick={() => indexNow.mutate()} disabled={indexNow.isPending}><RefreshCw className={cn('size-3', indexNow.isPending && 'animate-spin')} />{t('knowledge.indexNow')}</Button>
                    <Button size="sm" variant="ghost" onClick={() => reindex.mutate()} disabled={reindex.isPending}>{t('knowledge.reindex')}</Button>
                    <Button size="sm" variant="ghost" className="text-destructive" onClick={() => { if (window.confirm(t('knowledge.confirmDeleteBase', { name: base.name }))) deleteBase.mutate(base.slug) }}><Trash2 className="size-3" />{t('knowledge.delete')}</Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Main column */}
          <div className="space-y-6">
            {/* Ask */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">{t('knowledge.ask')}</CardTitle>
                <CardDescription>{t('knowledge.askHint')}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <form className="flex items-center gap-2" onSubmit={e => { e.preventDefault(); if (question.trim() && slug) ask.mutate(question.trim()) }}>
                  <Input value={question} onChange={e => setQuestion(e.target.value)} placeholder={t('knowledge.question')} />
                  <Button type="submit" disabled={!question.trim() || !slug || ask.isPending}><Send className="size-4" />{ask.isPending ? '…' : t('knowledge.ask')}</Button>
                </form>
                {answer && (
                  <div className="space-y-3 rounded-xl border p-4">
                    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <Badge variant={answer.status === 'ok' ? 'secondary' : 'destructive'}>{answer.status === 'ok' ? t('knowledge.answer') : t('knowledge.refused')}</Badge>
                      {answer.model && <span>{answer.model.platform}/{answer.model.modelId}</span>}
                      <span>{answer.latencyMs} ms</span>
                      <span>{t('knowledge.graphEngine')}: {answer.graph.engine}</span>
                      {answer.governance.citationsMissing && <Badge variant="destructive">{t('knowledge.citationsMissing')}</Badge>}
                      {answer.governance.redactions > 0 && <Badge variant="outline">{answer.governance.redactions} redactions</Badge>}
                    </div>
                    <p className="whitespace-pre-wrap text-sm">{answer.answer}</p>
                    {answer.citations.length > 0 && (
                      <div className="text-xs">
                        <div className="mb-1 font-medium">{t('knowledge.citations')}</div>
                        <ul className="space-y-0.5 text-muted-foreground">
                          {answer.citations.map(c => <li key={c.n}>[{c.n}] {c.title}{c.source ? ` — ${c.source}` : ''} · part {c.ordinal + 1}</li>)}
                        </ul>
                      </div>
                    )}
                    {answer.graph.nodes.length > 0 && (
                      <div className="flex flex-wrap gap-1 text-xs">
                        {answer.graph.nodes.slice(0, 12).map(n => <Badge key={n.id} variant="outline">{n.class}: {n.name}</Badge>)}
                      </div>
                    )}
                    <div className="flex items-center gap-3 text-xs">
                      <button type="button" className="underline text-muted-foreground" onClick={() => setShowSources(s => !s)}>
                        {showSources ? t('knowledge.hideSources') : t('knowledge.showSources', { count: answer.sources.length })}
                      </button>
                      <span className="text-muted-foreground">{t('knowledge.provenance')}: <code>{answer.id}</code></span>
                    </div>
                    {showSources && (
                      <ol className="space-y-2 text-xs">
                        {answer.sources.map((s, i) => (
                          <li key={s.chunkId} className="rounded-lg bg-muted/50 p-2">
                            <div className="mb-1 text-muted-foreground">[{i + 1}] {s.documentTitle} · part {s.ordinal + 1} · {s.score.toFixed(2)} · {s.via.join('+')}</div>
                            <div className="whitespace-pre-wrap">{s.text.slice(0, 600)}{s.text.length > 600 ? '…' : ''}</div>
                          </li>
                        ))}
                      </ol>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Documents */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">{t('knowledge.documents')}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <form
                  className="grid gap-2"
                  onSubmit={e => {
                    e.preventDefault()
                    if (!slug) return
                    if (docUrl.trim()) addDocument.mutate({ title: docTitle.trim() || undefined, url: docUrl.trim() })
                    else if (docText.trim()) addDocument.mutate({ title: docTitle.trim() || undefined, text: docText })
                  }}
                >
                  <div className="grid gap-2 sm:grid-cols-2">
                    <Input value={docTitle} onChange={e => setDocTitle(e.target.value)} placeholder={t('knowledge.docTitle')} />
                    <Input value={docUrl} onChange={e => setDocUrl(e.target.value)} placeholder={t('knowledge.docUrl')} />
                  </div>
                  <Textarea value={docText} onChange={e => setDocText(e.target.value)} placeholder={t('knowledge.docText')} rows={4} disabled={!!docUrl.trim()} />
                  <div>
                    <Button type="submit" size="sm" disabled={!slug || addDocument.isPending || (!docText.trim() && !docUrl.trim())}><Plus className="size-4" />{t('knowledge.addDocument')}</Button>
                  </div>
                </form>
                {docs.isLoading ? (
                  <Skeleton className="h-16 w-full" />
                ) : (docs.data?.documents.length ?? 0) === 0 ? (
                  <p className="text-sm text-muted-foreground">{t('knowledge.noDocs')}</p>
                ) : (
                  <ul className="divide-y text-sm">
                    {docs.data!.documents.map(d => (
                      <li key={d.id} className="flex items-center justify-between gap-3 py-2">
                        <div className="min-w-0">
                          <div className="truncate font-medium">{d.title}</div>
                          <div className="truncate text-xs text-muted-foreground">{d.source || '—'} · {d.chunkCount} {t('knowledge.chunks')}{d.redactions ? ` · ${d.redactions} redactions` : ''}{d.error ? ` · ${d.error}` : ''}</div>
                        </div>
                        <div className="flex shrink-0 items-center gap-2">
                          <span className={cn('rounded px-1.5 py-0.5 text-[10px] font-medium', STATUS_CLASS[d.status])}>{t(`knowledge.status_${d.status}`)}</span>
                          <Button size="icon" variant="ghost" aria-label={t('knowledge.delete')} onClick={() => deleteDocument.mutate(d.id)} disabled={deleteDocument.isPending}><Trash2 className="size-4" /></Button>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </CardContent>
            </Card>

            {/* Evals */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base"><FlaskConical className="mr-1 inline size-4" />{t('knowledge.evals')}</CardTitle>
                <CardDescription>{t('knowledge.evalsHint')}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex flex-wrap items-center gap-3">
                  <Select value={dataset} onValueChange={v => setDatasetChoice(v ?? '')}>
                    <SelectTrigger className="w-56"><SelectValue placeholder={t('knowledge.dataset')} /></SelectTrigger>
                    <SelectContent>
                      {(datasets.data?.datasets ?? []).map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                    </SelectContent>
                  </Select>
                  <label className="flex items-center gap-2 text-sm"><Switch checked={judge} onCheckedChange={setJudge} />{t('knowledge.judge')}</label>
                  <Button size="sm" onClick={() => runEval.mutate()} disabled={!slug || !dataset || runEval.isPending}>{runEval.isPending ? '…' : t('knowledge.run')}</Button>
                </div>
                {(evalRuns.data?.runs.length ?? 0) > 0 && (
                  <table className="w-full text-xs">
                    <thead className="text-left text-muted-foreground">
                      <tr><th className="py-1 pr-2">{t('knowledge.dataset')}</th><th className="pr-2">Cases</th><th className="pr-2">{t('knowledge.hitRate')}</th><th className="pr-2">MRR</th><th className="pr-2">{t('knowledge.correctness')}</th><th className="pr-2">{t('knowledge.faithfulness')}</th><th className="pr-2">{t('knowledge.refusalRate')}</th><th>p95</th></tr>
                    </thead>
                    <tbody>
                      {evalRuns.data!.runs.map(r => (
                        <tr key={r.id} className="border-t">
                          <td className="py-1 pr-2">{r.dataset} <span className="text-muted-foreground">{r.startedAt ? new Date(r.startedAt).toLocaleString() : ''}</span></td>
                          <td className="pr-2">{r.cases}</td>
                          <td className="pr-2">{pct(r.metrics.hitRate)}</td>
                          <td className="pr-2">{typeof r.metrics.mrr === 'number' ? r.metrics.mrr.toFixed(2) : '–'}</td>
                          <td className="pr-2">{pct(r.metrics.correctness)}</td>
                          <td className="pr-2">{pct(r.metrics.faithfulness)}</td>
                          <td className="pr-2">{pct(r.metrics.refusalRate)}</td>
                          <td>{typeof r.metrics.latencyP95Ms === 'number' ? `${r.metrics.latencyP95Ms} ms` : '–'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}

export default KnowledgePage
