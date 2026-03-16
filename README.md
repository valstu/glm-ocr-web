# GLM-OCR // On-Device Browser App

> Run [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) — a 0.9B multimodal OCR model — entirely in your browser via WebGPU. No server, no API key, no data leaves your device.

**[→ Live Demo](https://valstu.github.io/glm-ocr-web/)**

---

## What is GLM-OCR?

GLM-OCR is a 0.9B vision-language model by [Z.ai / Zhipu AI](https://www.z.ai/) that ranks **#1 on OmniDocBench v1.5 (94.62)**. It extracts structured content from documents:

| Task | Output |
|------|--------|
| Text Recognition | Plain text from images, photos, scans |
| Formula Recognition | LaTeX math |
| Table Recognition | HTML table |
| Document Parsing | Full Markdown with headings, lists, tables |
| Key Info Extraction | Structured JSON |

---

## How it works

```
User uploads image
      ↓
@huggingface/transformers (transformers.js v3)
      ↓
ONNX Runtime Web — WebGPU backend (GPU) or WASM (CPU fallback)
      ↓
GLM-OCR ONNX model (cached in browser after first download)
      ↓
Rendered output (Markdown / JSON / raw text)
```

**First visit**: model files are downloaded from HuggingFace Hub and cached by the browser.
**All subsequent visits**: inference runs 100% locally — no network needed.

---

## Getting started

### 1. Enable GitHub Pages

In your GitHub repo:
**Settings → Pages → Source → GitHub Actions**

Then trigger the deployment:
**Actions → "Deploy to GitHub Pages" → Run workflow**

The site will be live at `https://YOUR_USERNAME.github.io/glm-ocr-web/`

### 2. Convert the model to ONNX

The ONNX version of GLM-OCR needs to be created first. Use the included GitHub Actions workflow:

1. Add a `HF_TOKEN` secret to your repo
   *(Settings → Secrets and variables → Actions → New repository secret)*
   The token needs **write** access to HuggingFace Hub.

2. Run the conversion workflow:
   **Actions → "Convert GLM-OCR to ONNX" → Run workflow**

   | Input | Recommended value |
   |-------|-------------------|
   | `hf_repo_id` | `YOUR_HF_USERNAME/GLM-OCR-ONNX` |
   | `dtype` | `int4` |
   | `source_model` | `zai-org/GLM-OCR` |

3. Once the workflow finishes, update the **Model ID** field in the web app to `YOUR_HF_USERNAME/GLM-OCR-ONNX` and click **Load Model**.

### 3. Use the app

1. Click **Load Model** — model downloads and caches (~500 MB for INT4)
2. Drop in a document image
3. Pick a task: Text / Formula / Table / Document / KIE / Custom
4. Click **Run OCR**

---

## Running locally

```bash
git clone https://github.com/valstu/glm-ocr-web
cd glm-ocr-web
npm install
npm run dev
```

Open `http://localhost:5173`

---

## Converting the model yourself

If you'd rather run the conversion locally:

```bash
# Install deps
pip install "transformers>=5.3.0" torch onnx "onnxruntime>=1.20" "optimum[onnxruntime]"
pip install "huggingface_hub[hf_transfer]" Pillow numpy

# Convert + quantize to INT4
python conversion/convert_to_onnx.py \
  --output ./onnx_output \
  --quantize int4

# Push to HuggingFace Hub
python conversion/convert_to_onnx.py \
  --output ./onnx_output \
  --quantize int4 \
  --push_to_hub YOUR_USERNAME/GLM-OCR-ONNX
```

The script tries three approaches in order:
1. **optimum** — with a custom `GlmOcr` ONNX config (best for transformers.js)
2. **onnxruntime-genai** — Microsoft's model builder (KV-cache aware)
3. **torch.onnx.export** — direct fallback

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| UI | React 18 + TypeScript + Vite |
| ML runtime | `@huggingface/transformers` v3 (transformers.js) |
| Inference backend | ONNX Runtime Web — WebGPU / WASM |
| Styling | Custom CSS — dark terminal theme, JetBrains Mono |
| Deploy | GitHub Pages via GitHub Actions |
| Model conversion | Python — optimum / onnxruntime-genai / torch.onnx |

---

## Browser requirements

| Browser | WebGPU | WASM fallback |
|---------|--------|---------------|
| Chrome 113+ | ✓ | ✓ |
| Edge 113+ | ✓ | ✓ |
| Firefox | — | ✓ |
| Safari 18+ | ✓ | ✓ |

WebGPU is ~5–10× faster than WASM. If WebGPU isn't available, the app falls back to WASM automatically.

---

## Benchmarks (original PyTorch model)

| Benchmark | Score |
|-----------|-------|
| OmniDocBench v1.5 | **94.62 (#1)** |
| OCRBench | 94.0 |
| UniMERNet (formula) | 96.5 |
| ChartBench | SOTA |

---

## License

- **App code**: MIT
- **GLM-OCR model weights**: [MIT](https://huggingface.co/zai-org/GLM-OCR)
- **PP-DocLayoutV3** (used in conversion pipeline): Apache 2.0
