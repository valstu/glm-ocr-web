"""
GLM-OCR → ONNX Conversion Script
==================================
Converts zai-org/GLM-OCR to ONNX format compatible with transformers.js + WebGPU.

Tries three approaches in order:
  1. optimum with a custom GlmOcr ONNX config (best — transformers.js compatible)
  2. onnxruntime-genai model builder (good for on-device genai)
  3. Direct torch.onnx.export fallback (basic, may not support autoregressive generation)

Requirements (install with pip):
    pip install "transformers>=5.3.0" torch onnx "onnxruntime>=1.20" "optimum[onnxruntime]"
    pip install "huggingface_hub[hf_transfer]" Pillow numpy

Usage:
    # Basic conversion + INT4 quantize
    python convert_to_onnx.py --output ./onnx_out --quantize int4

    # Convert and push to HuggingFace Hub
    python convert_to_onnx.py --output ./onnx_out --quantize int4 \\
        --push_to_hub YOUR_USERNAME/GLM-OCR-ONNX
"""

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ── Helpers ────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    prefix = {"INFO": "[*]", "OK": "[+]", "WARN": "[!]", "ERR": "[✗]"}.get(level, "[?]")
    print(f"{prefix} {msg}", flush=True)


def fmt_mb(path: Path) -> str:
    if path.exists():
        return f"{path.stat().st_size / 1e6:.1f} MB"
    return "N/A"


def create_dummy_image(size: int = 448) -> Image.Image:
    return Image.fromarray(
        np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    )


def copy_tokenizer_files(src_dir: Path, dst_dir: Path):
    """Copy tokenizer, config and preprocessor files needed by transformers.js."""
    important = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ]
    for name in important:
        src = src_dir / name
        if src.exists():
            shutil.copy(src, dst_dir / name)
            log(f"Copied {name}", "OK")


def patch_config_for_transformers_js(config_path: Path):
    """Add hints that help transformers.js handle this model correctly."""
    if not config_path.exists():
        return
    with open(config_path) as f:
        cfg = json.load(f)
    cfg.setdefault("model_type", "glm_ocr")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)


# ── Approach 1: optimum with custom ONNX config ────────────────────────────

def try_optimum_export(
    model_id: str,
    output_dir: Path,
    dtype: str,
) -> bool:
    """
    Register a custom ONNX config for glm_ocr and use optimum to export.
    This produces files in the format transformers.js expects.
    """
    log("Approach 1: optimum with custom GlmOcr ONNX config")
    try:
        import torch
        from transformers import AutoConfig
        from optimum.exporters.onnx import main_export
        from optimum.exporters.onnx.config import (
            TextDecoderOnnxConfig,
            VisionOnnxConfig,
        )
        from optimum.exporters.tasks import TasksManager
        from optimum.utils import NormalizedTextConfig, NormalizedVisionConfig

        # ── Register custom ONNX config for glm_ocr ──
        log("Registering custom GlmOcr ONNX config...")

        class GlmOcrTextConfig(TextDecoderOnnxConfig):
            """Custom ONNX config for GLM-OCR's language decoder."""
            NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
            DEFAULT_ONNX_OPSET = 17

            @property
            def inputs(self):
                return {
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                }

            @property
            def outputs(self):
                return {"logits": {0: "batch_size", 1: "sequence_length"}}

        # Try to register with the tasks manager
        try:
            register_fn = TasksManager.create_register("onnx", overwrite_existing=True)
            register_fn("glm_ocr", "text-generation")(GlmOcrTextConfig)
            register_fn("glm_ocr", "image-text-to-text")(GlmOcrTextConfig)
            log("Custom config registered", "OK")
        except Exception as e:
            log(f"Registration warning (non-fatal): {e}", "WARN")

        # ── Export ──
        log(f"Running optimum export to {output_dir}...")
        dtype_map = {"int4": "int4", "int8": "int8", "fp16": "fp16", "fp32": "fp32"}
        onnx_dtype = dtype_map.get(dtype, "fp32")

        main_export(
            model_name_or_path=model_id,
            output=output_dir,
            task="image-text-to-text",
            opset=17,
            dtype=onnx_dtype if onnx_dtype in ("fp16", "fp32") else "fp32",
            trust_remote_code=True,
            library_name="transformers",
        )
        log("optimum export complete", "OK")

        # Post-quantize if needed
        if dtype in ("int4", "int8"):
            _quantize_onnx_files(output_dir, dtype)

        return True

    except Exception as e:
        log(f"optimum export failed: {e}", "WARN")
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        return False


def _quantize_onnx_files(onnx_dir: Path, dtype: str):
    """Quantize all ONNX files in a directory."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quant_type = QuantType.QInt4 if dtype == "int4" else QuantType.QInt8
        for onnx_file in onnx_dir.glob("*.onnx"):
            if "quantized" in onnx_file.name or "quant" in onnx_file.name:
                continue
            out_path = onnx_file.parent / f"{onnx_file.stem}_quantized.onnx"
            log(f"Quantizing {onnx_file.name} → {dtype}...")
            try:
                quantize_dynamic(str(onnx_file), str(out_path), weight_type=quant_type)
                log(f"Quantized: {out_path.name} ({fmt_mb(out_path)})", "OK")
            except Exception as e:
                log(f"Quantization of {onnx_file.name} failed: {e}", "WARN")
    except ImportError:
        log("onnxruntime.quantization not available, skipping quantization", "WARN")


# ── Approach 2: onnxruntime-genai model builder ────────────────────────────

def try_onnxruntime_genai(
    model_id: str,
    output_dir: Path,
    dtype: str,
) -> bool:
    """
    Use onnxruntime-genai's model builder — designed specifically for generative
    models with KV-cache. Produces ONNX + GenAI config files.
    """
    log("Approach 2: onnxruntime-genai model builder")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c", "import onnxruntime_genai"],
            capture_output=True,
        )
        if result.returncode != 0:
            log("onnxruntime-genai not installed, installing...", "WARN")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "onnxruntime-genai", "onnx-ir", "-q"],
                check=True,
            )

        precision_map = {
            "int4": "int4",
            "int8": "int8",
            "fp16": "fp16",
            "fp32": "fp32",
        }
        precision = precision_map.get(dtype, "int4")

        genai_output = output_dir / "genai"
        genai_output.mkdir(parents=True, exist_ok=True)

        log(f"Building onnxruntime-genai model (precision={precision})...")
        result = subprocess.run(
            [
                sys.executable, "-m", "onnxruntime_genai.models.builder",
                "-m", model_id,
                "-o", str(genai_output),
                "-p", precision,
                "-e", "cpu",
                "--trust_remote_code",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            log("onnxruntime-genai build complete", "OK")
            log(result.stdout[-2000:] if result.stdout else "", "INFO")
            return True
        else:
            log(f"onnxruntime-genai failed (exit {result.returncode}): {result.stderr[-1000:]}", "WARN")
            return False

    except Exception as e:
        log(f"onnxruntime-genai approach failed: {e}", "WARN")
        return False


# ── Approach 3: Direct torch.onnx.export ──────────────────────────────────

def try_direct_export(
    model_id: str,
    output_dir: Path,
    dtype: str,
) -> bool:
    """
    Export model components directly with torch.onnx.
    Exports vision encoder and the full model forward pass.
    """
    log("Approach 3: Direct torch.onnx.export")
    try:
        import torch
        from transformers import (
            AutoProcessor,
            AutoModelForImageTextToText,
            AutoConfig,
        )

        # Verify glm_ocr is known to this transformers install
        from transformers import CONFIG_MAPPING
        if "glm_ocr" not in CONFIG_MAPPING:
            log("glm_ocr not in CONFIG_MAPPING — transformers too old, skipping", "WARN")
            return False

        torch_dtype = torch.float16 if dtype in ("fp16", "int4", "int8") else torch.float32

        log(f"Loading model {model_id} (dtype={torch_dtype})...")

        # Load processor — try multiple strategies
        processor = None
        for kwargs in [
            {"trust_remote_code": True},
            {"trust_remote_code": False},
        ]:
            try:
                processor = AutoProcessor.from_pretrained(model_id, **kwargs)
                log(f"Processor loaded (trust_remote_code={kwargs['trust_remote_code']})", "OK")
                break
            except Exception as e:
                log(f"Processor load attempt failed: {e}", "WARN")

        if processor is None:
            log("Could not load processor", "ERR")
            return False

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()
        log(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)", "OK")

        onnx_dir = output_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        # ── Export vision encoder ──
        log("Exporting vision encoder...")
        vision_exported = _export_vision_encoder(model, onnx_dir, torch_dtype)

        # ── Export language model ──
        log("Exporting language model decoder...")
        lm_exported = _export_language_model(model, processor, onnx_dir, torch_dtype)

        if vision_exported or lm_exported:
            # Quantize if needed
            if dtype in ("int4", "int8"):
                _quantize_onnx_files(onnx_dir, dtype)
            return True
        return False

    except Exception as e:
        log(f"Direct export failed: {e}", "ERR")
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        return False


def _export_vision_encoder(model, onnx_dir: Path, torch_dtype) -> bool:
    """Export the CogViT visual encoder to ONNX."""
    import torch

    # Try to find the vision tower
    vision_module = None
    for attr in ["vision_tower", "visual", "vision_model", "image_encoder"]:
        m = getattr(model.model if hasattr(model, "model") else model, attr, None)
        if m is not None:
            vision_module = m
            log(f"Found vision module: model.{attr}")
            break

    if vision_module is None:
        # Look one level deeper
        if hasattr(model, "model"):
            for name, mod in model.model.named_children():
                if any(x in name.lower() for x in ["vision", "visual", "cog", "vit"]):
                    vision_module = mod
                    log(f"Found vision module: model.model.{name}")
                    break

    if vision_module is None:
        log("Could not locate vision encoder module", "WARN")
        return False

    # Also get the connector/projector
    connector = None
    for attr in ["multi_modal_projector", "mm_projector", "connector", "image_projection"]:
        c = getattr(model.model if hasattr(model, "model") else model, attr, None)
        if c is not None:
            connector = c
            log(f"Found connector: .{attr}")
            break

    class VisionEncoderWithConnector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = vision_module
            self.connector = connector

        def forward(self, pixel_values):
            out = self.vision(pixel_values)
            feat = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            if self.connector is not None:
                feat = self.connector(feat)
            return feat

    wrapper = VisionEncoderWithConnector().eval()
    dummy = torch.zeros(1, 3, 448, 448, dtype=torch_dtype)

    onnx_path = onnx_dir / "vision_encoder.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    log(f"Vision encoder: {onnx_path.name} ({fmt_mb(onnx_path)})", "OK")
    return True


def _export_language_model(model, processor, onnx_dir: Path, torch_dtype) -> bool:
    """Export the language model with a simple forward pass."""
    import torch

    class LMWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits

    try:
        wrapper = LMWrapper(model).eval()
        dummy_ids = torch.ones(1, 16, dtype=torch.long)
        dummy_mask = torch.ones(1, 16, dtype=torch.long)

        onnx_path = onnx_dir / "decoder_model.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_ids, dummy_mask),
                str(onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "logits": {0: "batch", 1: "seq"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
        log(f"Decoder model: {onnx_path.name} ({fmt_mb(onnx_path)})", "OK")
        return True
    except Exception as e:
        log(f"LM export failed: {e}", "WARN")
        return False


# ── Download tokenizer files ────────────────────────────────────────────────

def download_config_files(model_id: str, output_dir: Path) -> Path:
    """Download tokenizer and config files (not weights) from HF Hub."""
    log(f"Downloading tokenizer/config from {model_id}...")
    try:
        from huggingface_hub import snapshot_download
        cache = snapshot_download(
            model_id,
            ignore_patterns=["*.bin", "*.safetensors", "*.pt", "*.pth", "*.gguf"],
        )
        cache_path = Path(cache)
        copy_tokenizer_files(cache_path, output_dir)
        patch_config_for_transformers_js(output_dir / "config.json")
        return cache_path
    except Exception as e:
        log(f"Could not download config files: {e}", "WARN")
        return Path()


# ── Push to Hub ─────────────────────────────────────────────────────────────

def push_to_hub(output_dir: Path, repo_id: str):
    """Upload converted model to HuggingFace Hub."""
    log(f"Pushing to HuggingFace Hub: {repo_id}")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            repo_type="model",
        )
        log(f"Uploaded: https://huggingface.co/{repo_id}", "OK")
    except Exception as e:
        log(f"Hub push failed: {e}", "ERR")
        raise


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert GLM-OCR to ONNX for in-browser WebGPU inference"
    )
    parser.add_argument("--model",       default="zai-org/GLM-OCR")
    parser.add_argument("--output",      default="./onnx_output")
    parser.add_argument("--quantize",    default="int4", choices=["int4", "int8", "fp16", "fp32"])
    parser.add_argument("--push_to_hub", default=None,
                        help="HF repo ID to push to, e.g. username/GLM-OCR-ONNX")
    parser.add_argument("--approach",    default="auto",
                        choices=["auto", "optimum", "genai", "direct"],
                        help="Which export approach to use")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  GLM-OCR → ONNX Converter")
    print(f"  source : {args.model}")
    print(f"  output : {output_dir}")
    print(f"  dtype  : {args.quantize}")
    print("=" * 60)
    print()

    # Download tokenizer/config first (small, always works)
    download_config_files(args.model, output_dir)

    # Run conversion approaches
    success = False
    approaches = {
        "optimum": lambda: try_optimum_export(args.model, output_dir, args.quantize),
        "genai":   lambda: try_onnxruntime_genai(args.model, output_dir, args.quantize),
        "direct":  lambda: try_direct_export(args.model, output_dir, args.quantize),
    }

    if args.approach == "auto":
        order = ["optimum", "genai", "direct"]
    else:
        order = [args.approach]

    for name in order:
        log(f"\n{'─' * 50}")
        log(f"Trying: {name}")
        log(f"{'─' * 50}")
        try:
            if approaches[name]():
                success = True
                log(f"✓ Approach '{name}' succeeded!", "OK")
                break
            else:
                log(f"✗ Approach '{name}' produced no output", "WARN")
        except Exception as e:
            log(f"✗ Approach '{name}' crashed: {e}", "WARN")
            if os.environ.get("DEBUG"):
                traceback.print_exc()

    print()
    print("=" * 60)
    if success:
        # List output
        log("Conversion complete. Output files:", "OK")
        for f in sorted(output_dir.rglob("*")):
            if f.is_file():
                print(f"  {fmt_mb(f):>10}  {f.relative_to(output_dir)}")

        if args.push_to_hub:
            print()
            push_to_hub(output_dir, args.push_to_hub)
        else:
            print()
            log("To push to HuggingFace Hub, re-run with: --push_to_hub USERNAME/GLM-OCR-ONNX")
    else:
        log("All approaches failed. The model may need manual conversion.", "ERR")
        log("Options:", "ERR")
        log("  1. Open an issue on https://github.com/huggingface/optimum", "ERR")
        log("  2. Check onnxruntime-genai support for new model types", "ERR")
        log("  3. Set DEBUG=1 and re-run for full tracebacks", "ERR")
        sys.exit(1)

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
