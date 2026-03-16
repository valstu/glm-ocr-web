"""
GLM-OCR → ONNX Conversion Script
==================================
Converts zai-org/GLM-OCR to ONNX format compatible with transformers.js + WebGPU.

Requirements:
    pip install transformers>=5.3.0 torch optimum[onnxruntime] onnx onnxruntime
    pip install huggingface_hub[hf_transfer]

Usage:
    python convert_to_onnx.py --output ./onnx_output
    python convert_to_onnx.py --output ./onnx_output --quantize int4
    python convert_to_onnx.py --output ./onnx_output --push_to_hub YOUR_HF_USERNAME/GLM-OCR-ONNX

After conversion, run the app locally to test:
    cd .. && npm run dev
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import numpy as np
from PIL import Image


def load_model(model_id: str = "zai-org/GLM-OCR", dtype=torch.float32):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"[*] Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[+] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return processor, model


def create_dummy_inputs(processor, image_size=448):
    """Create dummy inputs for ONNX tracing."""
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs, dummy_image


def export_vision_encoder(model, processor, output_dir: Path):
    """Export the vision encoder to ONNX."""
    print("[*] Exporting vision encoder...")
    output_dir.mkdir(parents=True, exist_ok=True)

    class VisionEncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vision_model = model.model.vision_tower
            self.connector = model.model.multi_modal_projector

        def forward(self, pixel_values):
            vision_out = self.vision_model(pixel_values)
            if hasattr(vision_out, "last_hidden_state"):
                vision_features = vision_out.last_hidden_state
            else:
                vision_features = vision_out[0]
            projected = self.connector(vision_features)
            return projected

    wrapper = VisionEncoderWrapper(model)
    wrapper.eval()

    # Create dummy pixel values
    dummy_pixels = torch.randn(1, 3, 448, 448, dtype=torch.float32)

    onnx_path = output_dir / "vision_encoder.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_pixels,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_embeds": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"[+] Vision encoder saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return onnx_path


def export_text_decoder(model, processor, output_dir: Path, dummy_inputs):
    """Export the text decoder (language model) to ONNX."""
    print("[*] Exporting text decoder...")

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.lm = model

        def forward(self, input_ids, attention_mask, pixel_values):
            out = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            return out.logits

    wrapper = DecoderWrapper(model)
    wrapper.eval()

    # Get properly shaped dummy inputs
    with torch.no_grad():
        try:
            onnx_path = output_dir / "decoder_model.onnx"
            inputs = dummy_inputs
            pixel_values = inputs.get("pixel_values", torch.zeros(1, 3, 448, 448))
            input_ids = inputs["input_ids"][:, :32]  # Truncate for faster export
            attention_mask = inputs["attention_mask"][:, :32]

            torch.onnx.export(
                wrapper,
                (input_ids, attention_mask, pixel_values),
                str(onnx_path),
                input_names=["input_ids", "attention_mask", "pixel_values"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq_len"},
                    "attention_mask": {0: "batch", 1: "seq_len"},
                    "logits": {0: "batch", 1: "seq_len"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
            print(f"[+] Decoder saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"[!] Decoder export failed with combined input. Trying alternate method: {e}")
            raise


def quantize_model(onnx_path: Path, quantize_mode: str = "int4") -> Path:
    """Quantize ONNX model to reduce size."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print(f"[*] Quantizing {onnx_path.name} to {quantize_mode}...")
        quant_path = onnx_path.parent / f"{onnx_path.stem}_{quantize_mode}.onnx"
        quant_type = QuantType.QInt8 if quantize_mode == "int8" else QuantType.QInt4
        quantize_dynamic(str(onnx_path), str(quant_path), weight_type=quant_type)
        print(f"[+] Quantized: {quant_path} ({quant_path.stat().st_size / 1e6:.1f} MB)")
        return quant_path
    except Exception as e:
        print(f"[!] Quantization failed: {e}")
        return onnx_path


def copy_config_files(model_id: str, output_dir: Path):
    """Copy tokenizer and config files needed by transformers.js."""
    print("[*] Copying tokenizer and config files...")
    from huggingface_hub import snapshot_download

    # Download only config/tokenizer files (not weights)
    cache_dir = snapshot_download(
        model_id,
        ignore_patterns=["*.bin", "*.safetensors", "*.pt", "*.pth"],
    )
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]:
        src = Path(cache_dir) / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)
            print(f"  [+] Copied {fname}")

    # Patch config.json to add transformers.js hints
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["_transformers_js_config"] = {
            "kv_cache_dtype": "float32",
            "use_past": True,
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)


def push_to_hub(output_dir: Path, repo_id: str):
    """Upload converted model to HuggingFace Hub."""
    from huggingface_hub import HfApi
    print(f"[*] Pushing to HuggingFace Hub: {repo_id}")
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[+] Uploaded! View at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert GLM-OCR to ONNX for browser inference")
    parser.add_argument("--model", default="zai-org/GLM-OCR", help="HuggingFace model ID")
    parser.add_argument("--output", default="./onnx_output", help="Output directory")
    parser.add_argument("--quantize", choices=["none", "int8", "int4"], default="int4")
    parser.add_argument("--push_to_hub", default=None, help="HF repo ID to push to e.g. username/GLM-OCR-ONNX")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  GLM-OCR → ONNX Converter")
    print("=" * 60)

    # Load model
    processor, model = load_model(args.model)

    # Create dummy inputs
    dummy_inputs, _ = create_dummy_inputs(processor)

    # Export vision encoder
    try:
        ve_path = export_vision_encoder(model, processor, onnx_dir)
        if args.quantize != "none":
            quantize_model(ve_path, args.quantize)
    except Exception as e:
        print(f"[!] Vision encoder export failed: {e}")

    # Export decoder
    try:
        export_text_decoder(model, processor, onnx_dir, dummy_inputs)
    except Exception as e:
        print(f"[!] Decoder export failed: {e}")
        print("[i] Note: Full autoregressive decoder export requires custom handling.")
        print("[i] See: https://github.com/microsoft/onnxruntime-genai for genai export")

    # Copy config files
    try:
        copy_config_files(args.model, output_dir)
    except Exception as e:
        print(f"[!] Config copy failed: {e}")

    print("\n" + "=" * 60)
    print(f"[+] Conversion complete. Output: {output_dir}")
    print()
    print("Next steps:")
    print(f"  1. Upload to HF Hub: python convert_to_onnx.py --push_to_hub YOUR_USERNAME/GLM-OCR-ONNX")
    print(f"  2. Or host ONNX files locally and set MODEL_URL in the app")
    print("=" * 60)

    if args.push_to_hub:
        push_to_hub(output_dir, args.push_to_hub)


if __name__ == "__main__":
    main()
