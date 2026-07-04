import { useState, useRef, useEffect, useCallback, type DragEvent } from 'react';
import axios from 'axios';
import { Upload, Link2, FileText, CheckCircle2, XCircle, Loader2, RotateCcw, ChevronDown, ChevronUp, Clock, Zap } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const POLL_MS = 1500;

// ── Types ──────────────────────────────────────────────────────────────────────

type JobStatus = 'queued' | 'running' | 'done' | 'failed';

interface LogLine {
  ts: string;
  level: string;
  stage: number;
  msg: string;
}

interface JobResult {
  total_chunks: number;
  avg_score: number;
  used_fallback: boolean;
  segments: number;
}

interface Job {
  job_id: string;
  filename: string;
  status: JobStatus;
  log_lines: LogLine[];
  result: JobResult | null;
  error: string | null;
  created_at: string;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function logColor(level: string): string {
  switch (level) {
    case 'pass':
    case 'done':
      return 'var(--success)';
    case 'fail':
    case 'warn':
      return '#f97316';
    case 'info':
      return 'var(--accent-gold)';
    default:
      return 'var(--text-muted)';
  }
}

function stageLabel(stage: number): string {
  if (stage < 0) return 'ERR';
  if (stage === 0) return 'S0';
  return `S${stage}`;
}

function scoreBand(score: number): { label: string; color: string } {
  if (score >= 0.85) return { label: 'Excellent', color: 'var(--success)' };
  if (score >= 0.70) return { label: 'Good', color: '#4ade80' };
  if (score >= 0.55) return { label: 'Acceptable', color: '#f97316' };
  return { label: 'Poor', color: 'var(--danger)' };
}

// ── Sub-components ────────────────────────────────────────────────────────────

function LiveLogPanel({ lines }: { lines: LogLine[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [lines.length]);

  if (lines.length === 0) {
    return (
      <div style={{
        background: 'rgba(0,0,0,0.3)',
        borderRadius: 10,
        border: '1px solid var(--border-color)',
        padding: '20px',
        textAlign: 'center',
        color: 'var(--text-muted)',
        fontSize: 13,
        fontStyle: 'italic',
      }}>
        Pipeline logs will appear here…
      </div>
    );
  }

  return (
    <div ref={ref} style={{
      background: 'rgba(0,0,0,0.35)',
      borderRadius: 10,
      border: '1px solid var(--border-color)',
      padding: '12px 14px',
      fontFamily: 'monospace',
      fontSize: 12.5,
      lineHeight: 1.65,
      maxHeight: 320,
      overflowY: 'auto',
    }}>
      {lines.map((l, i) => (
        <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 2 }}>
          <span style={{ color: 'var(--text-muted)', minWidth: 64 }}>{l.ts}</span>
          <span style={{
            color: '#1a1a2e',
            background: logColor(l.level),
            borderRadius: 4,
            padding: '0 5px',
            fontSize: 10,
            fontWeight: 700,
            minWidth: 30,
            textAlign: 'center',
            alignSelf: 'flex-start',
            marginTop: 1,
          }}>{stageLabel(l.stage)}</span>
          <span style={{ color: logColor(l.level), flex: 1 }}>{l.msg}</span>
        </div>
      ))}
    </div>
  );
}

function ResultCard({ job }: { job: Job }) {
  const ok = job.status === 'done';
  const r = job.result;
  const band = r ? scoreBand(r.avg_score) : null;

  return (
    <div style={{
      background: ok
        ? 'rgba(16, 185, 129, 0.06)'
        : 'rgba(239, 68, 68, 0.06)',
      border: `1px solid ${ok ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`,
      borderRadius: 12,
      padding: '18px 20px',
      display: 'flex',
      flexDirection: 'column',
      gap: 14,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {ok
          ? <CheckCircle2 size={22} color="var(--success)" />
          : <XCircle size={22} color="var(--danger)" />}
        <span style={{ fontWeight: 700, fontSize: 15 }}>
          {ok ? 'Ingestion Complete' : 'Ingestion Failed'}
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: 12, marginLeft: 'auto' }}>
          {job.filename}
        </span>
      </div>

      {ok && r && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
          {[
            { label: 'Chunks Stored', value: r.total_chunks.toLocaleString() },
            { label: 'Avg Score', value: r.avg_score.toFixed(3) },
            { label: 'Quality', value: band?.label, color: band?.color },
            { label: 'Fallback Used', value: r.used_fallback ? 'Yes' : 'No', color: r.used_fallback ? '#f97316' : 'var(--success)' },
          ].map(stat => (
            <div key={stat.label} style={{
              background: 'rgba(255,255,255,0.04)',
              borderRadius: 8,
              padding: '10px 12px',
              textAlign: 'center',
            }}>
              <div style={{ color: stat.color || 'var(--text-main)', fontSize: 18, fontWeight: 700 }}>
                {stat.value}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 2 }}>
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      )}

      {!ok && job.error && (
        <div style={{
          background: 'rgba(239,68,68,0.08)',
          borderRadius: 8,
          padding: '10px 14px',
          color: 'var(--danger)',
          fontSize: 13,
          fontFamily: 'monospace',
        }}>
          {job.error}
        </div>
      )}
    </div>
  );
}

function JobHistoryRow({ job }: { job: Job }) {
  const [open, setOpen] = useState(false);
  const ok = job.status === 'done';
  const failed = job.status === 'failed';

  return (
    <div style={{
      border: '1px solid var(--border-color)',
      borderRadius: 10,
      overflow: 'hidden',
      marginBottom: 8,
    }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: '100%',
          background: 'rgba(255,255,255,0.03)',
          border: 'none',
          padding: '10px 14px',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          cursor: 'pointer',
          color: 'var(--text-main)',
        }}
      >
        {ok && <CheckCircle2 size={14} color="var(--success)" />}
        {failed && <XCircle size={14} color="var(--danger)" />}
        {!ok && !failed && <Loader2 size={14} color="var(--accent-gold)" className="spin" />}
        <span style={{ fontSize: 13, flex: 1, textAlign: 'left', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {job.filename}
        </span>
        {job.result && (
          <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>
            {job.result.total_chunks} chunks · {job.result.avg_score.toFixed(3)}
          </span>
        )}
        {open ? <ChevronUp size={14} color="var(--text-muted)" /> : <ChevronDown size={14} color="var(--text-muted)" />}
      </button>
      {open && (
        <div style={{ padding: '0 14px 14px' }}>
          <LiveLogPanel lines={job.log_lines} />
          {(ok || failed) && <div style={{ marginTop: 10 }}><ResultCard job={job} /></div>}
        </div>
      )}
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function IngestPage() {
  const [tab, setTab] = useState<'upload' | 'url'>('upload');
  const [dragging, setDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [urlValue, setUrlValue] = useState('');
  const [noLlm, setNoLlm] = useState(false);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [history, setHistory] = useState<Job[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [urlError, setUrlError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Polling ────────────────────────────────────────────────────────────────

  const startPolling = useCallback((jobId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get<Job>(`${API_URL}/api/ingest/status/${jobId}`);
        const job = res.data;
        setActiveJob(job);
        if (job.status === 'done' || job.status === 'failed') {
          clearInterval(pollRef.current!);
          pollRef.current = null;
          setHistory(prev => [job, ...prev.filter(j => j.job_id !== jobId)]);
        }
      } catch {
        // tolerate transient errors
      }
    }, POLL_MS);
  }, []);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // ── Submit Upload ──────────────────────────────────────────────────────────

  const handleUpload = async () => {
    if (!selectedFile) return;
    setSubmitting(true);
    setActiveJob(null);
    try {
      const form = new FormData();
      form.append('file', selectedFile);
      form.append('no_llm', String(noLlm));
      const res = await axios.post<{ job_id: string; filename: string; status: string }>(
        `${API_URL}/api/ingest/upload`, form, { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      const initial: Job = {
        job_id: res.data.job_id,
        filename: res.data.filename,
        status: 'queued',
        log_lines: [],
        result: null,
        error: null,
        created_at: new Date().toISOString(),
      };
      setActiveJob(initial);
      startPolling(res.data.job_id);
    } catch (e: unknown) {
      const msg = axios.isAxiosError(e) ? e.response?.data?.detail ?? e.message : String(e);
      setActiveJob({ job_id: '', filename: selectedFile.name, status: 'failed', log_lines: [], result: null, error: msg, created_at: new Date().toISOString() });
    } finally {
      setSubmitting(false);
    }
  };

  // ── Submit URL ─────────────────────────────────────────────────────────────

  const handleUrlIngest = async () => {
    setUrlError('');
    const url = urlValue.trim();
    if (!url) { setUrlError('Please enter a URL'); return; }
    setSubmitting(true);
    setActiveJob(null);
    try {
      const res = await axios.post<{ job_id: string; filename: string; status: string }>(
        `${API_URL}/api/ingest/url`, { url, no_llm: noLlm }
      );
      const initial: Job = {
        job_id: res.data.job_id,
        filename: res.data.filename,
        status: 'queued',
        log_lines: [],
        result: null,
        error: null,
        created_at: new Date().toISOString(),
      };
      setActiveJob(initial);
      startPolling(res.data.job_id);
    } catch (e: unknown) {
      const msg = axios.isAxiosError(e) ? e.response?.data?.detail ?? e.message : String(e);
      setUrlError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  // ── Drag and Drop ──────────────────────────────────────────────────────────

  const onDragOver = (e: DragEvent) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.name.toLowerCase().endsWith('.pdf')) setSelectedFile(f);
  };

  const isRunning = activeJob?.status === 'queued' || activeJob?.status === 'running';
  const isDone = activeJob?.status === 'done' || activeJob?.status === 'failed';

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      overflowY: 'auto',
      padding: '32px 40px',
      gap: 24,
      maxWidth: 800,
      margin: '0 auto',
      width: '100%',
    }}>

      {/* ── Header ──────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14, marginBottom: 4 }}>
        <div style={{
          width: 42, height: 42,
          background: 'linear-gradient(135deg, var(--accent-gold), #b8860b)',
          borderRadius: 10,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
        }}>
          <Upload size={20} color="#000" />
        </div>
        <div>
          <h2 style={{ fontSize: 22, marginBottom: 4, fontFamily: 'var(--font-serif)' }}>
            Ingest Legal Document
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: 13.5, lineHeight: 1.5 }}>
            Upload a PDF or paste a direct link. The pipeline will extract, chunk, and index it into ChromaDB automatically.
          </p>
        </div>
      </div>

      {/* ── Tab Toggle ──────────────────────────────────────────────── */}
      <div style={{
        display: 'flex',
        gap: 0,
        background: 'rgba(255,255,255,0.04)',
        borderRadius: 10,
        border: '1px solid var(--border-color)',
        padding: 4,
      }}>
        {(['upload', 'url'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              flex: 1,
              padding: '9px 0',
              borderRadius: 7,
              border: 'none',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: 13.5,
              transition: 'all 0.2s',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7,
              background: tab === t
                ? 'linear-gradient(135deg, var(--accent-gold), #b8860b)'
                : 'transparent',
              color: tab === t ? '#000' : 'var(--text-muted)',
            }}
          >
            {t === 'upload' ? <FileText size={15} /> : <Link2 size={15} />}
            {t === 'upload' ? 'Upload PDF' : 'PDF URL'}
          </button>
        ))}
      </div>

      {/* ── Upload Zone ─────────────────────────────────────────────── */}
      {tab === 'upload' && (
        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => !selectedFile && fileInputRef.current?.click()}
          style={{
            border: `2px dashed ${dragging ? 'var(--accent-gold)' : selectedFile ? 'rgba(16,185,129,0.5)' : 'var(--border-color)'}`,
            borderRadius: 14,
            padding: '36px 24px',
            textAlign: 'center',
            cursor: selectedFile ? 'default' : 'pointer',
            background: dragging
              ? 'rgba(212,175,55,0.06)'
              : selectedFile
                ? 'rgba(16,185,129,0.04)'
                : 'rgba(255,255,255,0.02)',
            transition: 'all 0.2s ease',
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            style={{ display: 'none' }}
            onChange={e => { if (e.target.files?.[0]) setSelectedFile(e.target.files[0]); }}
          />
          {selectedFile ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
              <div style={{
                width: 48, height: 48,
                background: 'rgba(16,185,129,0.15)',
                borderRadius: 12,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <FileText size={24} color="var(--success)" />
              </div>
              <div>
                <div style={{ fontWeight: 600, fontSize: 14 }}>{selectedFile.name}</div>
                <div style={{ color: 'var(--text-muted)', fontSize: 12, marginTop: 2 }}>
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </div>
              </div>
              <button
                className="btn-secondary"
                style={{ fontSize: 12, padding: '5px 14px', marginTop: 4 }}
                onClick={e => { e.stopPropagation(); setSelectedFile(null); if (fileInputRef.current) fileInputRef.current.value = ''; }}
              >
                Change File
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
              <div style={{
                width: 52, height: 52,
                background: 'rgba(212,175,55,0.1)',
                borderRadius: 14,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Upload size={24} color="var(--accent-gold)" />
              </div>
              <div>
                <div style={{ fontWeight: 600, color: 'var(--accent-gold)', fontSize: 14 }}>
                  {dragging ? 'Drop it here' : 'Drag & drop a PDF'}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: 12.5, marginTop: 3 }}>
                  or click to browse — only .pdf files accepted
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── URL Input ────────────────────────────────────────────────── */}
      {tab === 'url' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <label style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 500 }}>
            Direct PDF URL
          </label>
          <input
            type="url"
            placeholder="https://example.com/act.pdf"
            value={urlValue}
            onChange={e => { setUrlValue(e.target.value); setUrlError(''); }}
            style={{
              background: 'rgba(255,255,255,0.05)',
              border: `1px solid ${urlError ? 'var(--danger)' : 'var(--border-color)'}`,
              borderRadius: 10,
              padding: '12px 16px',
              color: 'var(--text-main)',
              fontSize: 14,
              width: '100%',
            }}
          />
          {urlError && (
            <div style={{ color: 'var(--danger)', fontSize: 12.5 }}>⚠ {urlError}</div>
          )}
          <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
            The server will download the PDF and save it to <code style={{ color: 'var(--accent-gold)' }}>data/legal_pdfs/</code>
          </div>
        </div>
      )}

      {/* ── Options + Ingest Button ──────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        {/* No-LLM toggle */}
        <label style={{
          display: 'flex', alignItems: 'center', gap: 9, cursor: 'pointer',
          fontSize: 13.5, color: 'var(--text-muted)', userSelect: 'none',
        }}>
          <div
            onClick={() => setNoLlm(!noLlm)}
            style={{
              width: 36, height: 20,
              borderRadius: 20,
              background: noLlm ? 'var(--accent-gold)' : 'rgba(255,255,255,0.12)',
              position: 'relative',
              cursor: 'pointer',
              transition: 'background 0.2s',
              flexShrink: 0,
            }}
          >
            <div style={{
              position: 'absolute',
              top: 2, left: noLlm ? 18 : 2,
              width: 16, height: 16,
              borderRadius: '50%',
              background: noLlm ? '#000' : 'rgba(255,255,255,0.6)',
              transition: 'left 0.2s',
            }} />
          </div>
          <Zap size={13} color={noLlm ? 'var(--accent-gold)' : 'var(--text-muted)'} />
          Regex-only (no LLM — faster)
        </label>

        <div style={{ flex: 1 }} />

        {isDone && (
          <button
            className="btn-secondary"
            style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={() => { setActiveJob(null); setSelectedFile(null); setUrlValue(''); }}
          >
            <RotateCcw size={14} />
            Ingest Another
          </button>
        )}

        <button
          className="btn-primary"
          disabled={isRunning || submitting || (tab === 'upload' ? !selectedFile : !urlValue.trim())}
          onClick={tab === 'upload' ? handleUpload : handleUrlIngest}
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            opacity: (isRunning || submitting) ? 0.7 : 1,
            cursor: (isRunning || submitting) ? 'not-allowed' : 'pointer',
            minWidth: 160,
            justifyContent: 'center',
          }}
        >
          {(isRunning || submitting)
            ? <><Loader2 size={16} className="spin" /> Running…</>
            : <><Upload size={16} /> Launch Pipeline</>}
        </button>
      </div>

      {/* ── Active Job Status Banner ─────────────────────────────────── */}
      {activeJob && (
        <div style={{
          background: 'var(--panel-bg)',
          border: '1px solid var(--border-color)',
          borderRadius: 14,
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
          backdropFilter: 'blur(12px)',
          animation: 'fadeIn 0.3s ease',
        }}>
          {/* Status header */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {isRunning && <Loader2 size={18} color="var(--accent-gold)" className="spin" />}
            {activeJob.status === 'queued' && <Clock size={18} color="var(--accent-gold)" />}
            {activeJob.status === 'done' && <CheckCircle2 size={18} color="var(--success)" />}
            {activeJob.status === 'failed' && <XCircle size={18} color="var(--danger)" />}
            <span style={{ fontWeight: 600, fontSize: 14 }}>
              {activeJob.status === 'queued' && 'Queued — waiting to start'}
              {activeJob.status === 'running' && 'Pipeline running…'}
              {activeJob.status === 'done' && 'Ingestion complete'}
              {activeJob.status === 'failed' && 'Ingestion failed'}
            </span>
            <code style={{ color: 'var(--text-muted)', fontSize: 11, marginLeft: 'auto' }}>
              {activeJob.filename}
            </code>
          </div>

          {/* Log panel */}
          <LiveLogPanel lines={activeJob.log_lines} />

          {/* Result card */}
          {isDone && <ResultCard job={activeJob} />}
        </div>
      )}

      {/* ── Job History ──────────────────────────────────────────────── */}
      {history.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <h4 style={{
            fontSize: 12,
            color: 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            fontFamily: 'var(--font-sans)',
          }}>
            Session History
          </h4>
          {history.map(job => <JobHistoryRow key={job.job_id} job={job} />)}
        </div>
      )}

      {/* ── Spin animation ───────────────────────────────────────────── */}
      <style>{`
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
