/**
 * RegulaStream — Real-Time Regulatory Intelligence Dashboard
 * =============================================================
 * A single-file React application (no extra runtime deps) that wires
 * every backend endpoint into a cohesive, premium UI.
 *
 * Layout
 * ──────
 *  ┌── Header ──────────────────────────────────────────────┐
 *  │   Logo · Connection Status · Live Clock                │
 *  ├── Main (3-column grid) ────────────────────────────────┤
 *  │  LEFT: Query Input + Pipeline Tracker                  │
 *  │  CENTER: Answer Panel (versioned, citations coloured)  │
 *  │  RIGHT: Regulatory Stream Monitor sidebar              │
 *  └────────────────────────────────────────────────────────┘
 *
 * Real API integration
 * ─────────────────────
 *  • WebSocket  /ws/stream            → live keystroke controller decisions
 *  • POST       /api/decomposer/decompose   → sub-query split
 *  • POST       /api/synthesizer/synthesize → answer generation
 *  • POST       /api/validator/validate     → citation audit
 *  • POST       /api/monitor/evaluate       → knowledge staleness check
 *  • POST       /api/session/init           → session state
 *  • POST       /api/session/delta_query    → late refinement
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

/* ── Constants ──────────────────────────────────────────────────────────── */
const API = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws/stream'

const STAGE_ORDER = ['controller', 'decomposer', 'synthesizer', 'validator']
const STAGE_LABELS = {
  controller:  'Retrieval Controller',
  decomposer:  'Intent Decomposer',
  synthesizer: 'RAG Synthesizer',
  validator:   'Citation Validator',
}
const STAGE_ICONS = {
  controller:  '⚡',
  decomposer:  '🔀',
  synthesizer: '🧠',
  validator:   '✅',
}

/* ── Utility ────────────────────────────────────────────────────────────── */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function post(path, body) {
  const res = await fetch(`${API}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

/** Extract [CITATION] patterns and colour them */
function ColourCitations({ text }) {
  if (!text) return null
  const parts = text.split(/(\[[A-Z0-9_§\s.,-]+\])/g)
  return parts.map((part, i) =>
    /^\[.+\]$/.test(part)
      ? <span key={i} className="citation-tag">{part}</span>
      : <span key={i}>{part}</span>
  )
}

/** Tiny animated spinner */
function Spinner() {
  return <span className="spinner" aria-label="Loading" />
}

/** Live pulsing dot */
function LiveDot({ active }) {
  return (
    <span className={`live-dot ${active ? 'live-dot--active' : ''}`}>
      <span className="live-dot__ring" />
    </span>
  )
}

/** Decision badge */
function DecisionBadge({ decision }) {
  const map = { RETRIEVE: 'badge--retrieve', WAIT: 'badge--wait', 'NO-RETRIEVAL': 'badge--no' }
  return <span className={`badge ${map[decision] || ''}`}>{decision || '—'}</span>
}

/* ── Pipeline Stage Card ────────────────────────────────────────────────── */
function StageCard({ id, status, latency, subQueries, citationResult }) {
  const states = { idle: 'idle', active: 'active', done: 'done', error: 'error' }
  return (
    <div className={`stage-card stage-card--${states[status] || 'idle'}`} id={`stage-${id}`}>
      <div className="stage-card__header">
        <span className="stage-card__icon">{STAGE_ICONS[id]}</span>
        <span className="stage-card__label">{STAGE_LABELS[id]}</span>
        {status === 'active' && <Spinner />}
        {status === 'done'   && <span className="stage-card__check">✓</span>}
        {status === 'error'  && <span className="stage-card__error-icon">✗</span>}
        {latency != null && <span className="stage-card__latency">{latency.toFixed(1)}ms</span>}
      </div>

      {/* Sub-query pills for decomposer */}
      {id === 'decomposer' && subQueries?.length > 0 && (
        <div className="stage-card__body">
          {subQueries.map((sq, i) => (
            <div key={i} className="sub-query-pill">
              <span className="sub-query-pill__intent">{sq.intent}</span>
              <span className="sub-query-pill__query">{sq.search_query}</span>
            </div>
          ))}
        </div>
      )}

      {/* Citation audit for validator */}
      {id === 'validator' && citationResult && (
        <div className="stage-card__body">
          <div className={`citation-status ${citationResult.is_valid ? 'citation-status--ok' : 'citation-status--fail'}`}>
            {citationResult.is_valid
              ? '✓ All citations verified'
              : `⚠ Fabricated IDs: ${citationResult.fabricated_ids_found?.join(', ')}`}
          </div>
        </div>
      )}
    </div>
  )
}

/* ── Regulatory Stream Event ────────────────────────────────────────────── */
function StreamEvent({ event, isNew }) {
  return (
    <div className={`stream-event ${event.update_required ? 'stream-event--alert' : 'stream-event--ok'} ${isNew ? 'stream-event--new' : ''}`}>
      <div className="stream-event__header">
        <span className="stream-event__doc-id">{event.doc_id}</span>
        <span className={`stream-event__badge ${event.update_required ? 'badge--retrieve' : 'badge--no'}`}>
          {event.update_required ? '⚠ UPDATE REQUIRED' : '✓ No Change'}
        </span>
      </div>
      {event.update_required && event.affected_claims?.length > 0 && (
        <ul className="stream-event__claims">
          {event.affected_claims.slice(0, 2).map((c, i) => (
            <li key={i} className="stream-event__claim">{c}</li>
          ))}
        </ul>
      )}
      <p className="stream-event__reasoning">{event.reasoning}</p>
    </div>
  )
}

/* ── Monitor Sidebar ────────────────────────────────────────────────────── */
function MonitorSidebar({ streamEvents, onInject }) {
  const [docId, setDocId] = useState('')
  const [docText, setDocText] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleInject(e) {
    e.preventDefault()
    if (!docId.trim() || !docText.trim()) return
    setLoading(true)
    try {
      await onInject(docId.trim(), docText.trim())
      setDocId('')
      setDocText('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <aside className="monitor-sidebar" aria-label="Regulatory Stream Monitor">
      <div className="monitor-sidebar__header">
        <span className="monitor-sidebar__icon">📡</span>
        <h2 className="monitor-sidebar__title">Regulatory Stream</h2>
        <LiveDot active={streamEvents.length > 0} />
      </div>

      {/* Inject a new document */}
      <form className="inject-form" onSubmit={handleInject}>
        <p className="inject-form__label">Inject Amendment</p>
        <input
          id="inject-doc-id"
          className="rs-input"
          placeholder="Document ID (e.g. REG_KYC_2026_AMEND)"
          value={docId}
          onChange={(e) => setDocId(e.target.value)}
        />
        <textarea
          id="inject-doc-text"
          className="rs-textarea rs-textarea--sm"
          placeholder="Paste the new regulatory text here…"
          value={docText}
          onChange={(e) => setDocText(e.target.value)}
          rows={4}
        />
        <button id="inject-submit-btn" type="submit" className="rs-btn rs-btn--warning" disabled={loading}>
          {loading ? <Spinner /> : '⚡ Inject & Evaluate'}
        </button>
      </form>

      {/* Event feed */}
      <div className="stream-feed" role="log" aria-live="polite" aria-label="Regulatory stream events">
        {streamEvents.length === 0 && (
          <p className="stream-feed__empty">No amendments injected yet. Inject a document above to see live staleness detection.</p>
        )}
        {streamEvents.map((ev, i) => (
          <StreamEvent key={ev.id} event={ev} isNew={i === 0} />
        ))}
      </div>
    </aside>
  )
}

/* ── Answer Panel ───────────────────────────────────────────────────────── */
function AnswerPanel({ answer, version, isValid, isLoading, isStale }) {
  return (
    <section className={`answer-panel ${isStale ? 'answer-panel--stale' : ''}`} aria-label="Generated answer">
      <div className="answer-panel__header">
        <div className="answer-panel__title-row">
          <h2 className="answer-panel__title">Answer</h2>
          {version > 0 && (
            <span className="version-badge">v{version}</span>
          )}
          {isStale && (
            <span className="stale-badge">⚠ Knowledge Updated</span>
          )}
        </div>
        {isValid != null && (
          <div className={`citation-validity ${isValid ? 'citation-validity--ok' : 'citation-validity--fail'}`}>
            {isValid
              ? <><span>✓</span> Citations verified</>
              : <><span>⚠</span> Hallucinated citations detected</>
            }
          </div>
        )}
      </div>

      <div className="answer-panel__body">
        {isLoading && (
          <div className="answer-panel__loading">
            <Spinner />
            <span>Synthesizing answer…</span>
          </div>
        )}
        {!isLoading && !answer && (
          <p className="answer-panel__placeholder">
            Your compliance answer will appear here after you submit a query.
          </p>
        )}
        {!isLoading && answer && (
          <div className="answer-text">
            <ColourCitations text={answer} />
          </div>
        )}
      </div>
    </section>
  )
}

/* ── Main App ────────────────────────────────────────────────────────────── */
export default function App() {
  /* ── State ── */
  const [query, setQuery] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [wsStatus, setWsStatus] = useState('disconnected') // 'connected' | 'disconnected'

  // Controller (WebSocket)
  const [controllerDecision, setControllerDecision] = useState(null)
  const [controllerConfidence, setControllerConfidence] = useState(null)
  const [controllerLatency, setControllerLatency] = useState(null)

  // Pipeline stages
  const [stages, setStages] = useState({
    controller:  { status: 'idle', latency: null },
    decomposer:  { status: 'idle', latency: null, subQueries: [] },
    synthesizer: { status: 'idle', latency: null },
    validator:   { status: 'idle', latency: null, citationResult: null },
  })

  // Answer
  const [answer, setAnswer] = useState('')
  const [answerVersion, setAnswerVersion] = useState(0)
  const [citationValid, setCitationValid] = useState(null)

  // Session
  const [sessionId, setSessionId] = useState(null)
  const [isAnswerStale, setIsAnswerStale] = useState(false)

  // Monitor stream
  const [streamEvents, setStreamEvents] = useState([])

  // Late refinement bar
  const [refinementQuery, setRefinementQuery] = useState('')
  const [isRefining, setIsRefining] = useState(false)

  // Evidence cache (for monitor & refinement)
  const lastEvidenceRef = useRef([])
  const wsRef = useRef(null)
  const debounceRef = useRef(null)
  const eventIdRef = useRef(0)

  /* ── WebSocket: live controller feedback while typing ─────────────────── */
  useEffect(() => {
    function connect() {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => setWsStatus('connected')
      ws.onclose = () => {
        setWsStatus('disconnected')
        setTimeout(connect, 3000) // auto-reconnect
      }
      ws.onerror = () => ws.close()

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data)
          setControllerDecision(data.decision)
          setControllerConfidence(data.confidence)
          setControllerLatency(data.latency_ms)
          setStages((prev) => ({
            ...prev,
            controller: { status: 'done', latency: data.latency_ms },
          }))
        } catch {}
      }
    }
    connect()
    return () => wsRef.current?.close()
  }, [])

  /* ── Live keystroke → controller via WebSocket ──────────────────────── */
  const sendToController = useCallback((text) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      wsRef.current.send(JSON.stringify({ chunk: text, timestamp: Date.now() / 1000 }))
      setStages((prev) => ({
        ...prev,
        controller: { ...prev.controller, status: 'active' },
      }))
    }, 180) // debounce 180ms
  }, [])

  function handleQueryChange(e) {
    const val = e.target.value
    setQuery(val)
    if (val.trim()) sendToController(val)
  }

  /* ── Full pipeline submission ─────────────────────────────────────────── */
  async function runPipeline(queryText, isRefinement = false) {
    setIsSubmitting(true)
    setIsAnswerStale(false)

    // Reset stages to idle → active sequence
    setStages({
      controller:  { status: 'done',   latency: controllerLatency },
      decomposer:  { status: 'active',  latency: null, subQueries: [] },
      synthesizer: { status: 'idle',    latency: null },
      validator:   { status: 'idle',    latency: null, citationResult: null },
    })
    setCitationValid(null)

    try {
      /* 1. Decompose ──────────────────────────────────────────────────── */
      const decompResult = await post('/api/decomposer/decompose', { query: queryText })
      setStages((prev) => ({
        ...prev,
        decomposer: {
          status: 'done',
          latency: decompResult.latency_ms,
          subQueries: decompResult.sub_queries || [],
        },
        synthesizer: { status: 'active', latency: null },
      }))

      /* 2. Build evidence chunks from sub-query text (simulated retrieval) */
      // In production this would query a vector store. We use the sub-query
      // descriptions themselves as evidence chunks to demonstrate the flow.
      const chunks = (decompResult.sub_queries || [{ intent: 'main', search_query: queryText }]).map(
        (sq, i) => ({
          doc_id: `SUB_${String(i + 1).padStart(3, '0')}`,
          text: sq.search_query,
        })
      )
      // Carry over cached evidence for refinements
      if (isRefinement && lastEvidenceRef.current.length > 0) {
        const existingIds = new Set(lastEvidenceRef.current.map((e) => e.doc_id))
        for (const c of chunks) {
          if (!existingIds.has(c.doc_id)) lastEvidenceRef.current.push(c)
        }
      } else {
        lastEvidenceRef.current = chunks
      }

      /* 3. Synthesize ─────────────────────────────────────────────────── */
      const prevSid = sessionId
      const historyStrings = prevSid ? [`Q: ${queryText} → A: (cached)`] : []

      const synthResult = await post('/api/synthesizer/synthesize', {
        current_query:   queryText,
        retrieved_chunks: lastEvidenceRef.current,
        session_history: historyStrings,
      })

      const generatedAnswer = synthResult.answer_markdown || synthResult.answer || '(No answer generated)'
      setAnswer(generatedAnswer)
      setAnswerVersion((v) => v + 1)

      setStages((prev) => ({
        ...prev,
        synthesizer: { status: 'done', latency: synthResult.latency_ms },
        validator: { status: 'active', latency: null },
      }))

      /* 4. Validate citations ────────────────────────────────────────── */
      const validIds = lastEvidenceRef.current.map((c) => c.doc_id)
      const valResult = await post('/api/validator/validate', {
        generated_answer:   generatedAnswer,
        valid_document_ids: validIds,
      })

      setCitationValid(valResult.is_valid)
      setStages((prev) => ({
        ...prev,
        validator: {
          status: valResult.is_valid ? 'done' : 'error',
          latency: valResult.latency_ms,
          citationResult: valResult,
        },
      }))

      /* 5. Init / update session ─────────────────────────────────────── */
      const newSid = prevSid || `rs-${Date.now()}`
      await post('/api/session/init', {
        session_id: newSid,
        query:      queryText,
        answer:     generatedAnswer,
        evidence:   lastEvidenceRef.current.map((c) => ({ doc_id: c.doc_id, text: c.text })),
      })
      setSessionId(newSid)

    } catch (err) {
      console.error('[RegulaStream] Pipeline error:', err)
      // Mark last active stage as error
      setStages((prev) => {
        const updated = { ...prev }
        for (const id of STAGE_ORDER.slice().reverse()) {
          if (updated[id].status === 'active') {
            updated[id] = { ...updated[id], status: 'error' }
            break
          }
        }
        return updated
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  function handleSubmit(e) {
    e.preventDefault()
    if (!query.trim() || isSubmitting) return
    runPipeline(query.trim(), false)
  }

  /* ── Late Refinement ────────────────────────────────────────────────── */
  async function handleRefinement(e) {
    e.preventDefault()
    if (!refinementQuery.trim() || !sessionId || isRefining) return
    setIsRefining(true)
    try {
      // Get delta query
      const deltaResult = await post('/api/session/delta_query', {
        session_id:     sessionId,
        new_constraint: refinementQuery.trim(),
      })

      // Run pipeline with combined constraint
      const combinedQuery = deltaResult.search_query || refinementQuery.trim()
      await runPipeline(combinedQuery, true)
      setRefinementQuery('')
    } catch (err) {
      console.error('[RegulaStream] Refinement error:', err)
    } finally {
      setIsRefining(false)
    }
  }

  /* ── Monitor Injection ──────────────────────────────────────────────── */
  async function handleMonitorInject(docId, docText) {
    if (!answer) return
    const result = await post('/api/monitor/evaluate', {
      previous_query:   query,
      previous_answer:  answer,
      new_doc_id:       docId,
      new_document_text: docText,
    })

    const id = ++eventIdRef.current
    setStreamEvents((prev) => [
      { id, doc_id: docId, ...result },
      ...prev.slice(0, 19),
    ])

    if (result.update_required) {
      setIsAnswerStale(true)
    }
  }

  /* ── Confidence bar ─────────────────────────────────────────────────── */
  const confPct = controllerConfidence != null ? Math.round(controllerConfidence * 100) : 0

  /* ── UI ──────────────────────────────────────────────────────────────── */
  return (
    <div className="app">
      {/* ── Header ──────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="app-header__brand">
          <span className="app-header__logo">⚖️</span>
          <div>
            <h1 className="app-header__title">RegulaStream</h1>
            <p className="app-header__subtitle">Real-Time Regulatory Intelligence</p>
          </div>
        </div>
        <div className="app-header__status">
          <div className={`connection-badge ${wsStatus === 'connected' ? 'connection-badge--ok' : 'connection-badge--off'}`}>
            <LiveDot active={wsStatus === 'connected'} />
            <span>{wsStatus === 'connected' ? 'Live' : 'Offline'}</span>
          </div>
          <span className="app-header__api">API: {API}</span>
        </div>
      </header>

      {/* ── Main Grid ───────────────────────────────────────────────── */}
      <main className="app-main">

        {/* ── LEFT: Query + Pipeline ──────────────────────────────── */}
        <div className="left-panel">

          {/* Query Input */}
          <section className="query-section" aria-label="Query input">
            <h2 className="panel-heading">Query</h2>
            <form onSubmit={handleSubmit} className="query-form">
              <div className="query-input-wrap">
                <textarea
                  id="query-textarea"
                  className="rs-textarea"
                  placeholder="Ask a regulatory compliance question… (e.g. What are the KYC requirements for digital lending under the latest RBI circular?)"
                  value={query}
                  onChange={handleQueryChange}
                  rows={4}
                  disabled={isSubmitting}
                />
                {/* Live controller feedback */}
                {controllerDecision && (
                  <div className="controller-live">
                    <DecisionBadge decision={controllerDecision} />
                    {controllerLatency != null && (
                      <span className="controller-live__latency">{controllerLatency.toFixed(1)}ms</span>
                    )}
                    {controllerConfidence != null && (
                      <div className="confidence-bar" role="meter" aria-valuenow={confPct} aria-valuemin="0" aria-valuemax="100">
                        <div className="confidence-bar__fill" style={{ '--w': `${confPct}%`, width: `${confPct}%` }} />
                        <span className="confidence-bar__label">{confPct}%</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
              <button
                id="submit-query-btn"
                type="submit"
                className="rs-btn rs-btn--primary"
                disabled={isSubmitting || !query.trim()}
              >
                {isSubmitting ? <><Spinner /> Running Pipeline…</> : '▶ Run Pipeline'}
              </button>
            </form>

            {/* Late Refinement */}
            {sessionId && (
              <form onSubmit={handleRefinement} className="refinement-form">
                <p className="refinement-form__label">
                  <span className="badge badge--info">G5</span> Late Refinement
                </p>
                <div className="refinement-row">
                  <input
                    id="refinement-input"
                    className="rs-input"
                    placeholder="Add a constraint (e.g. …only for NRI customers)"
                    value={refinementQuery}
                    onChange={(e) => setRefinementQuery(e.target.value)}
                    disabled={isRefining}
                  />
                  <button
                    id="refinement-submit-btn"
                    type="submit"
                    className="rs-btn rs-btn--secondary"
                    disabled={isRefining || !refinementQuery.trim()}
                  >
                    {isRefining ? <Spinner /> : '↩ Refine'}
                  </button>
                </div>
              </form>
            )}
          </section>

          {/* Pipeline Tracker */}
          <section className="pipeline-section" aria-label="Pipeline tracker">
            <h2 className="panel-heading">
              Pipeline Tracker
              <span className="badge badge--info" style={{ marginLeft: 8 }}>G6 Trace</span>
            </h2>
            <div className="pipeline-stages">
              {STAGE_ORDER.map((id, i) => (
                <div key={id} className="pipeline-stage-wrap">
                  <StageCard
                    id={id}
                    status={stages[id]?.status}
                    latency={stages[id]?.latency}
                    subQueries={stages[id]?.subQueries}
                    citationResult={stages[id]?.citationResult}
                  />
                  {i < STAGE_ORDER.length - 1 && (
                    <div className={`pipeline-connector ${stages[STAGE_ORDER[i + 1]]?.status !== 'idle' ? 'pipeline-connector--active' : ''}`} />
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* ── CENTER: Answer Panel ─────────────────────────────────── */}
        <div className="center-panel">
          <AnswerPanel
            answer={answer}
            version={answerVersion}
            isValid={citationValid}
            isLoading={isSubmitting || isRefining}
            isStale={isAnswerStale}
          />
        </div>

        {/* ── RIGHT: Regulatory Stream ─────────────────────────────── */}
        <MonitorSidebar
          streamEvents={streamEvents}
          onInject={handleMonitorInject}
        />
      </main>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="app-footer">
        <span>RegulaStream · Samsung PRISM Hackathon · Gates G1–G6</span>
        {sessionId && <span className="app-footer__session">Session: <code>{sessionId}</code></span>}
      </footer>
    </div>
  )
}
