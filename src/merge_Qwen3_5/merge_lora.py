#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PRETRAINED_MODEL_PATH = Path(
    "/home/txhung/MLOps/llm-ocr-product-workload/Resources/models/Qwen3.5-0.8B"
)
LORA_ADAPTER_PATH = Path(
    "/home/txhung/MLOps/llm-ocr-product-workload/artifacts/practice/"
    "practice-20261006-164826/dpo/adapter"
)
OUTPUT_MODEL_PATH = Path(
    "/media/txhung/xuanhung/models/finetuned_models/LLM/Qwen3.5_0.8B"
)


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"{label} directory does not exist: {path}")


def _validate_vllm_weight_keys(output_dir: Path) -> None:
    from safetensors import safe_open

    keys = set()
    for weight_file in output_dir.glob("model*.safetensors"):
        with safe_open(str(weight_file), framework="pt", device="cpu") as weights:
            keys.update(weights.keys())

    invalid_prefixes = (
        "model.language_model.language_model.",
        "model.language_model.visual.",
    )
    if any(key.startswith(invalid_prefixes) for key in keys):
        raise RuntimeError(
            "Saved Qwen3.5 weights have nested key prefixes that vLLM cannot load. "
            "Run the merge with the pinned requirements.txt environment."
        )
    if not any(key.startswith("model.language_model.layers.") for key in keys):
        raise RuntimeError("Saved weights are missing canonical Qwen3.5 language keys")
    if not any(key.startswith("model.visual.") for key in keys):
        raise RuntimeError("Saved weights are missing canonical Qwen3.5 vision keys")


def merge_lora(
    pretrained_model_path: Path,
    lora_adapter_path: Path,
    output_dir: Path,
) -> Path:
    """Merge the final local Qwen3.5 LoRA adapter into its base model."""
    _require_directory(pretrained_model_path, "Pretrained model")
    _require_directory(lora_adapter_path, "LoRA adapter")

    resolved_output = output_dir.resolve()
    for source in (pretrained_model_path, lora_adapter_path):
        resolved_source = source.resolve()
        if (
            resolved_output.is_relative_to(resolved_source)
            or resolved_source.is_relative_to(resolved_output)
        ):
            raise ValueError("Output directory must be separate from both inputs")

    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")

    import torch
    from peft import PeftConfig, PeftModel
    # Keep the save path on the Transformers version pinned alongside vLLM.
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    adapter_config = PeftConfig.from_pretrained(
        lora_adapter_path,
        local_files_only=True,
    )
    adapter_base = getattr(adapter_config, "base_model_name_or_path", None)
    if adapter_base and Path(adapter_base).resolve() != pretrained_model_path.resolve():
        raise ValueError(
            "Adapter was trained from a different base model: "
            f"{adapter_base} (configured: {pretrained_model_path})"
        )

    print(f"Loading processor: {lora_adapter_path}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(
        lora_adapter_path,
        local_files_only=True,
    )

    print(f"Loading pretrained model: {pretrained_model_path}", file=sys.stderr)
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        pretrained_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    print(f"Loading LoRA adapter: {lora_adapter_path}", file=sys.stderr)
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        is_trainable=False,
        local_files_only=True,
        config=adapter_config,
    )

    print("Merging LoRA weights into the pretrained model...", file=sys.stderr)
    # PEFT's standalone-model workflow: https://huggingface.co/docs/peft/en/developer_guides/checkpoint#merge-the-weights
    merged_model = peft_model.merge_and_unload(safe_merge=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model: {output_dir}", file=sys.stderr)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    # Saves the processor and tokenizer together: https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/processors#save_pretrained
    processor.save_pretrained(output_dir)

    if not any(output_dir.glob("model*.safetensors")):
        raise RuntimeError(f"Merged safetensors weights were not saved: {output_dir}")
    for name in ("config.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (output_dir / name).is_file():
            raise RuntimeError(f"Required inference file was not saved: {name}")
    _validate_vllm_weight_keys(output_dir)

    print("Merge complete.", file=sys.stderr)
    return output_dir


def main() -> int:
    merge_lora(
        PRETRAINED_MODEL_PATH,
        LORA_ADAPTER_PATH,
        OUTPUT_MODEL_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
