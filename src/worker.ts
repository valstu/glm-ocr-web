/**
 * GLM-OCR inference worker
 * Runs in a Web Worker to keep the UI thread responsive.
 * Uses @huggingface/transformers with WebGPU backend.
 */
import {
  env,
  AutoProcessor,
  AutoModelForImageTextToText,
  RawImage,
} from '@huggingface/transformers';

// Model ID — update this when an ONNX version is published
const MODEL_ID = 'onnx-community/GLM-OCR';

// Configure transformers.js to use HF Hub
env.allowRemoteModels = true;
env.allowLocalModels = false;

// ── Types ──────────────────────────────────────────────────────────────────
export interface WorkerRequest {
  type: 'load' | 'run';
  imageData?: string;
  prompt?: string;
  modelId?: string;
  device?: 'webgpu' | 'wasm';
  dtype?: string;
}

export interface WorkerResponse {
  type: 'progress' | 'ready' | 'result' | 'error' | 'log';
  text?: string;
  progress?: { file: string; loaded: number; total: number };
  error?: string;
  level?: 'info' | 'warn' | 'error' | 'ok';
}

// ── State ──────────────────────────────────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let processor: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let model: any = null;
let currentModelId: string | null = null;

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

function log(text: string, level: WorkerResponse['level'] = 'info') {
  post({ type: 'log', text, level });
}

// ── Load ───────────────────────────────────────────────────────────────────
async function loadModel(modelId: string, device: string, dtype: string) {
  if (currentModelId === modelId) {
    log('Model already loaded', 'ok');
    post({ type: 'ready' });
    return;
  }

  log(`Loading model: ${modelId}`, 'info');
  log(`Backend: ${device.toUpperCase()} | dtype: ${dtype}`, 'info');

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const progressCallback = (progress: any) => {
    if (progress.status === 'downloading' || progress.status === 'progress') {
      post({
        type: 'progress',
        progress: {
          file: progress.file || progress.name || '',
          loaded: progress.loaded || 0,
          total: progress.total || 0,
        },
      });
    } else if (progress.status === 'loading') {
      log(`Loading: ${progress.file || ''}`, 'info');
    } else if (progress.status === 'done') {
      log(`Ready: ${progress.file || modelId}`, 'ok');
    }
  };

  try {
    processor = await AutoProcessor.from_pretrained(modelId, {
      progress_callback: progressCallback,
    });

    model = await AutoModelForImageTextToText.from_pretrained(modelId, {
      device: device as 'webgpu' | 'wasm',
      dtype: dtype as 'fp32',
      progress_callback: progressCallback,
    });

    currentModelId = modelId;
    log(`Model loaded successfully on ${device.toUpperCase()}`, 'ok');
    post({ type: 'ready' });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    log(`Failed to load model: ${msg}`, 'error');
    if (msg.includes('404') || msg.includes('not found') || msg.includes('does not exist')) {
      post({ type: 'error', error: `MODEL_NOT_FOUND:${modelId}` });
    } else {
      post({ type: 'error', error: msg });
    }
  }
}

// ── Run ────────────────────────────────────────────────────────────────────
async function runInference(imageDataUrl: string, prompt: string) {
  if (!processor || !model) {
    post({ type: 'error', error: 'Model not loaded. Click Load Model first.' });
    return;
  }

  log(`Running inference: "${prompt}"`, 'info');

  try {
    const image = await RawImage.fromURL(imageDataUrl);

    const messages = [
      {
        role: 'user',
        content: [
          { type: 'image', image },
          { type: 'text', text: prompt },
        ],
      },
    ];

    const inputs = await processor.apply_chat_template(messages, {
      tokenize: true,
      add_generation_prompt: true,
      return_dict: true,
      return_tensors: 'pt',
    });

    log('Generating output...', 'info');

    const generated = await model.generate({
      ...inputs,
      max_new_tokens: 4096,
    });

    const inputLen: number = inputs.input_ids?.dims?.[1] ?? 0;
    // generated may be a tensor or array-like; decode from output tokens
    const outputTokens = Array.isArray(generated[0])
      ? generated[0].slice(inputLen)
      : generated[0];

    const decoded: string = processor.decode(outputTokens, { skip_special_tokens: true });

    log('Inference complete', 'ok');
    post({ type: 'result', text: decoded });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    log(`Inference error: ${msg}`, 'error');
    post({ type: 'error', error: msg });
  }
}

// ── Message Handler ────────────────────────────────────────────────────────
self.addEventListener('message', async (event: MessageEvent<WorkerRequest>) => {
  const { type, imageData, prompt, modelId, device = 'webgpu', dtype = 'q4' } = event.data;

  if (type === 'load') {
    await loadModel(modelId || MODEL_ID, device, dtype);
  } else if (type === 'run') {
    if (!imageData || !prompt) {
      post({ type: 'error', error: 'Missing imageData or prompt' });
      return;
    }
    await runInference(imageData, prompt);
  }
});
