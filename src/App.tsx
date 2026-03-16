import { useState, useRef, useCallback, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

// ── Types ──────────────────────────────────────────────────────────────────
type TaskType = 'text' | 'formula' | 'table' | 'document' | 'kie' | 'custom';
type LogLevel = 'info' | 'warn' | 'error' | 'ok';
type AppPhase = 'idle' | 'loading-model' | 'ready' | 'running' | 'done' | 'error';

interface LogEntry {
  id: number;
  time: string;
  level: LogLevel;
  msg: string;
}

interface FileProgress {
  file: string;
  loaded: number;
  total: number;
}

interface ImageData {
  file: File;
  dataUrl: string;
}

const TASKS: { id: TaskType; icon: string; label: string; prompt: string }[] = [
  { id: 'text',     icon: '¶', label: 'Text',      prompt: 'Text Recognition:' },
  { id: 'formula',  icon: '∑', label: 'Formula',   prompt: 'Formula Recognition:' },
  { id: 'table',    icon: '⊞', label: 'Table',     prompt: 'Table Recognition:' },
  { id: 'document', icon: '§', label: 'Document',  prompt: 'Document Parsing:' },
  { id: 'kie',      icon: '{}', label: 'Extract',  prompt: 'Key Information Extraction:' },
  { id: 'custom',   icon: '>', label: 'Custom',    prompt: '' },
];

const MODEL_ID = 'onnx-community/GLM-OCR';
const DEVICE_OPTIONS = [
  { id: 'webgpu', label: 'WebGPU (GPU)' },
  { id: 'wasm',   label: 'WASM (CPU)' },
] as const;
const DTYPE_OPTIONS = ['q4', 'q4f16', 'q8', 'fp16', 'fp32'] as const;

let logCounter = 0;

function timestamp() {
  return new Date().toISOString().slice(11, 23);
}

function formatBytes(bytes: number) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

// ── Component ──────────────────────────────────────────────────────────────
export default function App() {
  const [phase, setPhase] = useState<AppPhase>('idle');
  const [image, setImage] = useState<ImageData | null>(null);
  const [task, setTask] = useState<TaskType>('document');
  const [customPrompt, setCustomPrompt] = useState('');
  const [kieSchema, setKieSchema] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [fileProgress, setFileProgress] = useState<Map<string, FileProgress>>(new Map());
  const [modelId, setModelId] = useState(MODEL_ID);
  const [device, setDevice] = useState<'webgpu' | 'wasm'>('webgpu');
  const [dtype, setDtype] = useState<string>('q4');
  const [modelNotFound, setModelNotFound] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [copied, setCopied] = useState(false);
  const [resultTab, setResultTab] = useState<'rendered' | 'raw'>('rendered');
  const [webgpuSupport, setWebgpuSupport] = useState<'checking' | 'yes' | 'no'>('checking');

  const workerRef = useRef<Worker | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  // ── WebGPU check ────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        if (!('gpu' in navigator)) { setWebgpuSupport('no'); return; }
        const adapter = await (navigator as unknown as { gpu: { requestAdapter: () => Promise<unknown> } }).gpu.requestAdapter();
        setWebgpuSupport(adapter ? 'yes' : 'no');
        if (!adapter) setDevice('wasm');
      } catch {
        setWebgpuSupport('no');
        setDevice('wasm');
      }
    })();
  }, []);

  // ── Worker setup ────────────────────────────────────────────────────────
  useEffect(() => {
    const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;

    w.onmessage = (e) => {
      const { type, text, progress, error: err, level } = e.data;

      if (type === 'log') {
        addLog(text, level ?? 'info');
      } else if (type === 'progress' && progress) {
        setFileProgress(prev => {
          const next = new Map(prev);
          next.set(progress.file, progress);
          return next;
        });
      } else if (type === 'ready') {
        setPhase('ready');
        addLog('Model ready — upload an image and run OCR', 'ok');
        setFileProgress(new Map());
      } else if (type === 'result') {
        setResult(text ?? '');
        setPhase('done');
        addLog('Result received', 'ok');
      } else if (type === 'error') {
        if (err?.startsWith('MODEL_NOT_FOUND:')) {
          setModelNotFound(true);
          setPhase('error');
          setError(`ONNX model not yet available at:\n${err.replace('MODEL_NOT_FOUND:', '')}\n\nRun conversion/convert_to_onnx.py then upload to HuggingFace Hub.`);
          addLog('ONNX model not found — see setup instructions', 'error');
        } else {
          setPhase('error');
          setError(err ?? 'Unknown error');
          addLog(err ?? 'Unknown error', 'error');
        }
      }
    };

    return () => w.terminate();
  }, []);

  // ── Auto-scroll logs ────────────────────────────────────────────────────
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const addLog = useCallback((msg: string, level: LogLevel = 'info') => {
    setLogs(prev => [...prev.slice(-199), {
      id: ++logCounter,
      time: timestamp(),
      level,
      msg,
    }]);
  }, []);

  // ── Image handling ───────────────────────────────────────────────────────
  const handleFiles = useCallback((files: FileList | null) => {
    if (!files?.length) return;
    const file = files[0];
    if (!file.type.startsWith('image/')) {
      addLog(`Not an image: ${file.name}`, 'warn');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      setImage({ file, dataUrl: e.target?.result as string });
      setResult(null);
      setError(null);
      setPhase(prev => prev === 'idle' ? 'idle' : prev);
      addLog(`Image loaded: ${file.name} (${formatBytes(file.size)})`, 'info');
    };
    reader.readAsDataURL(file);
  }, [addLog]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  // ── Load model ───────────────────────────────────────────────────────────
  const loadModel = useCallback(() => {
    if (!workerRef.current) return;
    setPhase('loading-model');
    setFileProgress(new Map());
    setModelNotFound(false);
    setError(null);
    addLog(`Loading ${modelId} on ${device} (${dtype})...`, 'info');
    workerRef.current.postMessage({ type: 'load', modelId, device, dtype });
  }, [modelId, device, dtype, addLog]);

  // ── Run OCR ──────────────────────────────────────────────────────────────
  const runOcr = useCallback(() => {
    if (!workerRef.current || !image) return;
    const taskDef = TASKS.find(t => t.id === task)!;
    let prompt = task === 'custom' ? customPrompt : taskDef.prompt;
    if (task === 'kie' && kieSchema.trim()) {
      prompt = `Key Information Extraction:\n${kieSchema.trim()}`;
    }
    if (!prompt.trim()) {
      addLog('Please enter a prompt', 'warn');
      return;
    }

    setPhase('running');
    setResult(null);
    setError(null);
    addLog(`Sending to worker: "${prompt}"`, 'info');
    workerRef.current.postMessage({ type: 'run', imageData: image.dataUrl, prompt });
  }, [image, task, customPrompt, kieSchema, addLog]);

  const copyResult = useCallback(() => {
    if (!result) return;
    navigator.clipboard.writeText(result).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [result]);

  // ── Render ───────────────────────────────────────────────────────────────
  const canLoad = phase === 'idle' || phase === 'error' || phase === 'done';
  const canRun  = (phase === 'ready' || phase === 'done') && !!image;
  const isRunning = phase === 'running' || phase === 'loading-model';

  const totalProgress = (() => {
    if (fileProgress.size === 0) return 0;
    let loaded = 0, total = 0;
    fileProgress.forEach(p => { loaded += p.loaded; total += p.total; });
    return total > 0 ? (loaded / total) * 100 : 0;
  })();

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-top">
          <div className="header-badge">
            <div className="badge-dot" />
            <span>GLM-OCR // ONNX WebGPU Runtime</span>
          </div>
          <div className="header-stats">
            <div className="stat">
              <span className="stat-label">MODEL</span>
              <span className="stat-value">0.9B</span>
            </div>
            <div className="stat">
              <span className="stat-label">OMNI</span>
              <span className="stat-value">94.62</span>
            </div>
            <div className="stat">
              <span className="stat-label">OCRBENCH</span>
              <span className="stat-value">94.0</span>
            </div>
            <div className="stat">
              <span className="stat-label">BACKEND</span>
              <span className="stat-value">{device.toUpperCase()}</span>
            </div>
            <div className="stat">
              {webgpuSupport === 'yes'
                ? <span className="webgpu-badge ok">⚡ WebGPU</span>
                : webgpuSupport === 'no'
                ? <span className="webgpu-badge warn">⚠ No WebGPU → WASM</span>
                : <span className="webgpu-badge ok">… checking</span>
              }
            </div>
          </div>
        </div>

        <div className="header-main">
          <pre className="ascii-logo">{`
 ██████╗ ██╗     ███╗   ███╗      ██████╗  ██████╗██████╗
██╔════╝ ██║     ████╗ ████║     ██╔═══██╗██╔════╝██╔══██╗
██║  ███╗██║     ██╔████╔██║     ██║   ██║██║     ██████╔╝
██║   ██║██║     ██║╚██╔╝██║     ██║   ██║██║     ██╔══██╗
╚██████╔╝███████╗██║ ╚═╝ ██║     ╚██████╔╝╚██████╗██║  ██║
 ╚═════╝ ╚══════╝╚═╝     ╚═╝      ╚═════╝  ╚═════╝╚═╝  ╚═╝`.trim()}</pre>

          <div className="header-meta">
            <div className="header-subtitle">on-device document intelligence</div>
            <div className="header-benchmarks">
              <div className="bench-row">
                <span className="bench-name">OmniDocBench v1.5</span>
                <span className="bench-score">#1 → 94.62</span>
              </div>
              <div className="bench-row">
                <span className="bench-name">OCRBench</span>
                <span className="bench-score">94.0</span>
              </div>
              <div className="bench-row">
                <span className="bench-name">UniMERNet (formula)</span>
                <span className="bench-score">96.5</span>
              </div>
              <div className="bench-row">
                <span className="bench-name">Runtime</span>
                <span className="bench-score">100% in-browser</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main>
        {/* LEFT — Controls */}
        <aside className="left-panel">
          {/* Image Drop */}
          <div className="dropzone-section">
            <div className="panel-title">
              <div className="panel-title-icon" />
              INPUT IMAGE
            </div>
            {image ? (
              <div className="image-preview-wrap">
                <img src={image.dataUrl} alt="input" className="image-preview" />
                <div className="image-preview-bar">
                  <span className="text-dim">{image.file.name}</span>
                  <button className="image-clear-btn" onClick={() => setImage(null)}>
                    [clear]
                  </button>
                </div>
              </div>
            ) : (
              <div
                className={`dropzone${dragOver ? ' drag-over' : ''}`}
                onDrop={onDrop}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onClick={() => fileInputRef.current?.click()}
              >
                <span className="dropzone-icon">📄</span>
                <div className="dropzone-text">Drop image here or click to upload</div>
                <div className="dropzone-hint">PNG, JPG, WEBP, PDF screenshot</div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="dropzone-input"
                  onChange={e => handleFiles(e.target.files)}
                />
              </div>
            )}
          </div>

          {/* Task */}
          <div className="controls-section">
            <div>
              <div className="field-label">OCR Task</div>
              <div className="task-grid">
                {TASKS.map(t => (
                  <button
                    key={t.id}
                    className={`task-btn${task === t.id ? ' active' : ''}`}
                    onClick={() => setTask(t.id)}
                  >
                    <span className="task-btn-icon">{t.icon}</span>
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            {task === 'kie' && (
              <div>
                <div className="field-label">JSON Schema (optional)</div>
                <textarea
                  className="input-field"
                  placeholder={'{\n  "name": "",\n  "date": "",\n  "total": ""\n}'}
                  value={kieSchema}
                  onChange={e => setKieSchema(e.target.value)}
                />
              </div>
            )}

            {task === 'custom' && (
              <div>
                <div className="field-label">Custom Prompt</div>
                <input
                  type="text"
                  className="input-field"
                  placeholder="e.g. Describe the document layout"
                  value={customPrompt}
                  onChange={e => setCustomPrompt(e.target.value)}
                />
              </div>
            )}
          </div>

          {/* Model Config */}
          <div className="model-section">
            <div className="panel-title">
              <div className="panel-title-icon" />
              MODEL CONFIG
            </div>
            <div className="controls-section">
              <div className="model-info">
                <div className="model-id">{modelId}</div>
                <div className="model-note">
                  ONNX version — runs entirely in your browser via WebGPU.
                  Model is cached after first download.
                  <br />
                  <a
                    href="https://github.com/valstu/glm-ocr-web/blob/main/conversion/convert_to_onnx.py"
                    target="_blank" rel="noreferrer"
                  >
                    [conversion script]
                  </a>
                </div>
              </div>

              <div>
                <div className="field-label">Model ID</div>
                <input
                  type="text"
                  className="input-field"
                  value={modelId}
                  onChange={e => setModelId(e.target.value)}
                  placeholder="username/GLM-OCR-ONNX"
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <div>
                  <div className="field-label">Backend</div>
                  <select
                    className="input-field"
                    value={device}
                    onChange={e => setDevice(e.target.value as 'webgpu' | 'wasm')}
                  >
                    {DEVICE_OPTIONS.map(d => (
                      <option key={d.id} value={d.id}>{d.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <div className="field-label">Precision</div>
                  <select
                    className="input-field"
                    value={dtype}
                    onChange={e => setDtype(e.target.value)}
                  >
                    {DTYPE_OPTIONS.map(d => (
                      <option key={d} value={d}>{d}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Buttons */}
          <div className="run-section">
            {canLoad && (
              <button className="run-btn" style={{ marginBottom: 8 }} onClick={loadModel}>
                {phase === 'idle' ? '[ LOAD MODEL ]' : '[ RELOAD MODEL ]'}
              </button>
            )}
            {phase === 'loading-model' && (
              <button className="run-btn loading" disabled>
                [ LOADING... {totalProgress.toFixed(0)}% ]
              </button>
            )}
            {(phase === 'ready' || phase === 'done') && (
              <button
                className="run-btn"
                onClick={runOcr}
                disabled={!canRun}
              >
                [ RUN OCR ]
              </button>
            )}
          </div>
        </aside>

        {/* RIGHT — Results */}
        <section className="right-panel">
          <div className="tab-bar">
            <div className="panel-title" style={{ flex: 1, border: 'none' }}>
              <div className="panel-title-icon" />
              OUTPUT
            </div>
            {result && (
              <>
                <button
                  className={`tab${resultTab === 'rendered' ? ' active' : ''}`}
                  onClick={() => setResultTab('rendered')}
                >RENDERED</button>
                <button
                  className={`tab${resultTab === 'raw' ? ' active' : ''}`}
                  onClick={() => setResultTab('raw')}
                >RAW</button>
              </>
            )}
          </div>

          <div className="results-area">
            {/* ── Loading model progress ── */}
            {phase === 'loading-model' && fileProgress.size > 0 && (
              <div className="model-load-panel">
                <div className="model-load-title">
                  ↓ Downloading model — cached after first run
                </div>
                <div style={{ marginBottom: 12 }}>
                  <div className="progress-bar-wrap" style={{ width: '100%' }}>
                    <div className="progress-bar-fill" style={{ width: `${totalProgress}%` }} />
                  </div>
                  <div className="progress-text" style={{ marginTop: 4 }}>
                    {totalProgress.toFixed(1)}% complete
                  </div>
                </div>
                <div className="file-progress-list">
                  {Array.from(fileProgress.values()).map(fp => (
                    <div key={fp.file} className="file-progress-item">
                      <span className="file-name">{fp.file.split('/').pop()}</span>
                      <span className="file-size">
                        {formatBytes(fp.loaded)} / {fp.total ? formatBytes(fp.total) : '?'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Loading spinner ── */}
            {phase === 'loading-model' && fileProgress.size === 0 && (
              <div className="loading-state">
                <div className="loading-spinner" />
                <div className="loading-label">INITIALIZING MODEL...</div>
              </div>
            )}

            {/* ── Running ── */}
            {phase === 'running' && (
              <div className="loading-state">
                <div className="loading-spinner" />
                <div className="loading-label">RUNNING OCR...</div>
              </div>
            )}

            {/* ── Model not found — show setup guide ── */}
            {modelNotFound && (
              <div className="setup-guide">
                <div className="setup-title">⚠ ONNX Model Not Found</div>
                <p style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 12, lineHeight: 1.6 }}>
                  The ONNX version of GLM-OCR isn't on HuggingFace Hub yet.
                  Run the included conversion script to create it:
                </p>
                <div className="setup-step">
                  <div className="step-num">1</div>
                  <div>
                    <div>Install dependencies</div>
                    <code className="setup-code">pip install transformers&gt;=5.3.0 torch optimum[onnxruntime] onnx</code>
                  </div>
                </div>
                <div className="setup-step">
                  <div className="step-num">2</div>
                  <div>
                    <div>Run conversion</div>
                    <code className="setup-code">python conversion/convert_to_onnx.py --output ./onnx_out --quantize int4</code>
                  </div>
                </div>
                <div className="setup-step">
                  <div className="step-num">3</div>
                  <div>
                    <div>Upload to HuggingFace Hub</div>
                    <code className="setup-code">python conversion/convert_to_onnx.py --push_to_hub YOUR_USERNAME/GLM-OCR-ONNX</code>
                  </div>
                </div>
                <div className="setup-step">
                  <div className="step-num">4</div>
                  <div>Set Model ID above to <span className="text-cyan">YOUR_USERNAME/GLM-OCR-ONNX</span> and reload</div>
                </div>
              </div>
            )}

            {/* ── Error ── */}
            {phase === 'error' && error && !modelNotFound && (
              <div className="error-box">
                <div className="error-title">⚡ ERROR</div>
                {error}
              </div>
            )}

            {/* ── Result ── */}
            {result && (
              <div>
                <div className="result-header">
                  <div className="result-label">
                    {TASKS.find(t => t.id === task)?.label ?? 'Output'}
                  </div>
                  <div className="result-actions">
                    <button className={`action-btn${copied ? ' copied' : ''}`} onClick={copyResult}>
                      {copied ? '[copied!]' : '[copy]'}
                    </button>
                  </div>
                </div>

                {resultTab === 'rendered' ? (
                  <div className="markdown-result">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeRaw]}
                    >
                      {result}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <pre className="raw-result">{result}</pre>
                )}
              </div>
            )}

            {/* ── Empty state ── */}
            {phase === 'idle' && !result && !error && (
              <div className="empty-state">
                <div className="empty-icon">🔬</div>
                <div className="empty-title">GLM-OCR // On-Device</div>
                <div className="empty-sub">
                  Load the model, upload a document image, and run OCR — 100% in your browser using WebGPU.
                  No data leaves your device.
                </div>
              </div>
            )}

            {phase === 'ready' && !result && (
              <div className="empty-state">
                <div className="empty-icon">✓</div>
                <div className="empty-title text-green">MODEL READY</div>
                <div className="empty-sub">
                  Upload an image and click RUN OCR
                </div>
              </div>
            )}
          </div>
        </section>
      </main>

      {/* ── Log Panel ── */}
      <footer className="log-panel">
        <div className="log-header">
          <span>TERMINAL LOG</span>
          <button className="log-clear" onClick={() => setLogs([])}>
            [clear]
          </button>
        </div>
        <div className="log-entries">
          {logs.map(entry => (
            <div key={entry.id} className="log-entry">
              <span className="log-time">{entry.time}</span>
              <span className={`log-level ${entry.level}`}>{entry.level.toUpperCase()}</span>
              <span className="log-msg">{entry.msg}</span>
            </div>
          ))}
          <div ref={logEndRef} />
        </div>
      </footer>
    </div>
  );
}
