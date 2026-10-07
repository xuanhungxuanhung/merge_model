#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


PRETRAINED_MODEL_PATH = Path(
    "/media/txhung/xuanhung/models/pretrained_models/pretrained_model"
)
LORA_ADAPTER_PATH = Path(
    "/media/txhung/xuanhung/models/finetuned_models/Hunyuan/adaptor/hunyuan_4ViT_projector_decoder_3epochs_v2/checkpoint-28000"
)
OUTPUT_MODEL_PATH = Path(
    "/media/txhung/xuanhung/models/finetuned_models/Hunyuan/merged_models/hunyuan_4ViT_projector_decoder_3epochs_v2"
)

_ADAPTER_KEY_MAPPING = {r"^model\.language_model\.": "model."}
_NON_LORA_WEIGHTS_NAME = "non_lora_trainables.safetensors"
_TRAINING_POLICY_NAME = "training_policy.json"


def _preserve_model_config(pretrained_model_path: Path, output_dir: Path) -> None:
    """Keep the source config format expected by runtimes such as vLLM.

    ``save_pretrained`` serializes the in-memory Transformers config. With
    Transformers 5 this rewrites Hunyuan's legacy ``rope_scaling`` /
    ``xdrope_section`` to ``rope_parameters`` / ``mrope_section``. vLLM
    needs an explicit ``rope_type=xdrope`` to build xD-RoPE positions.
    """
    source = pretrained_model_path / "config.json"
    target = output_dir / "config.json"
    if not source.is_file():
        raise FileNotFoundError(f"Pretrained model config is missing: {source}")

    config = json.loads(source.read_text(encoding="utf-8"))

    def normalize_xdrope(config_section: dict[str, Any]) -> None:
        rope_scaling = config_section.get("rope_scaling")
        if (
            not isinstance(rope_scaling, dict)
            or rope_scaling.get("type") != "xdrope"
        ):
            return
        rope_parameters = dict(rope_scaling)
        rope_parameters["rope_type"] = "xdrope"
        config_section["rope_scaling"] = rope_parameters
        config_section["rope_parameters"] = dict(rope_parameters)

    normalize_xdrope(config)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        normalize_xdrope(text_config)

    target.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _normalize_target_name(target: str) -> str:
    if target.startswith("model.language_model."):
        return target.replace("model.language_model.", "model.", 1)
    return target.replace(r"model\.language_model\.", r"model\.", 1)


def _normalize_target_modules(target_modules: Any) -> Any:
    """Map training-time decoder paths to the current Transformers layout."""
    if isinstance(target_modules, str):
        return _normalize_target_name(target_modules)
    if isinstance(target_modules, set):
        return {_normalize_target_name(target) for target in target_modules}
    if isinstance(target_modules, frozenset):
        return frozenset(_normalize_target_name(target) for target in target_modules)
    if isinstance(target_modules, list):
        return [_normalize_target_name(target) for target in target_modules]
    if isinstance(target_modules, tuple):
        return tuple(_normalize_target_name(target) for target in target_modules)
    raise TypeError(
        "LoRA target_modules must be a string or a collection of strings; "
        f"got {type(target_modules).__name__}"
    )


def _select_target_modules(model: Any, target_modules: Any) -> Any:
    """Choose the target-module spelling that exists in the loaded base model."""
    module_names = {name for name, _ in model.named_modules()}
    normalized = _normalize_target_modules(target_modules)
    candidates = [target_modules]
    if normalized != target_modules:
        candidates.append(normalized)

    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                pattern = re.compile(candidate)
            except re.error:
                continue
            if any(pattern.fullmatch(name) for name in module_names):
                return candidate
        elif isinstance(candidate, (set, frozenset, list, tuple)):
            if all(name in module_names for name in candidate):
                return candidate
    raise ValueError(
        "LoRA target_modules do not match any modules in the loaded base model"
    )


def _normalize_non_lora_key(source_key: str, model: Any | None = None) -> str:
    """Map a hybrid tensor using the nested or flat loaded-model topology."""
    nested_key = source_key.removeprefix("base_model.model.")
    if model is not None and nested_key in _model_tensors(model):
        return nested_key

    projector_prefix = "base_model.model.model.vision_tower.patch_merger."
    if source_key.startswith(projector_prefix):
        suffix = source_key.removeprefix(projector_prefix)
        if suffix.startswith("proj_conv."):
            suffix = suffix.replace("proj_conv.", "proj.0.", 1)
        elif suffix.startswith("proj_out."):
            suffix = suffix.replace("proj_out.", "proj.2.", 1)
        return f"vit.perceive.{suffix}"

    vision_layer_prefix = "base_model.model.model.vision_tower.layers."
    if source_key.startswith(vision_layer_prefix):
        suffix = source_key.removeprefix(vision_layer_prefix)
        suffix = suffix.replace(".layer_norm1.", ".input_layernorm.", 1)
        suffix = suffix.replace(".layer_norm2.", ".post_attention_layernorm.", 1)
        suffix = suffix.replace(".mlp.fc1.", ".mlp.dense_h_to_4h.", 1)
        suffix = suffix.replace(".mlp.fc2.", ".mlp.dense_4h_to_h.", 1)
        return f"vit.layers.{suffix}"

    raise ValueError(f"Unsupported non-LoRA tensor: {source_key}")


def _model_tensors(model: Any) -> dict[str, Any]:
    tensors = dict(model.named_parameters())
    for name, tensor in model.named_buffers():
        if name in tensors:
            raise ValueError(f"Model exposes duplicate tensor name: {name}")
        tensors[name] = tensor
    return tensors


def _load_non_lora_trainables(model: Any, adapter_dir: Path) -> tuple[str, ...]:
    """Load and strictly validate the non-LoRA sidecar from a hybrid checkpoint."""
    weights_path = adapter_dir / _NON_LORA_WEIGHTS_NAME
    manifest_path = adapter_dir / _TRAINING_POLICY_NAME
    if not weights_path.exists() and not manifest_path.exists():
        return ()
    if not weights_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Hybrid checkpoint requires both "
            f"{_NON_LORA_WEIGHTS_NAME} and {_TRAINING_POLICY_NAME}"
        )

    from safetensors.torch import load_file

    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    schema_version = manifest.get("schema_version")
    source_names = manifest.get(
        "persisted_non_lora_parameter_names"
        if schema_version == 2
        else "non_lora_trainable_names"
    )
    if (
        schema_version not in (1, 2)
        or manifest.get("decoder_mode") != "lora"
        or not isinstance(source_names, list)
        or not all(isinstance(name, str) for name in source_names)
        or len(source_names) != len(set(source_names))
    ):
        raise ValueError(f"Invalid hybrid checkpoint manifest: {manifest_path}")

    source_state = load_file(str(weights_path), device="cpu")
    if set(source_state) != set(source_names):
        raise ValueError(
            "Hybrid checkpoint manifest does not match non-LoRA tensor keys"
        )

    available = _model_tensors(model)
    normalized: dict[str, Any] = {}
    for source_name, source_tensor in source_state.items():
        target_name = _normalize_non_lora_key(source_name, model)
        if target_name in normalized:
            raise ValueError(f"Duplicate normalized non-LoRA tensor: {target_name}")
        target_tensor = available.get(target_name)
        if target_tensor is None:
            raise ValueError(
                f"Non-LoRA tensor target is absent from the base model: {target_name}"
            )
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"Non-LoRA tensor shape mismatch for {target_name}: "
                f"checkpoint={tuple(source_tensor.shape)}, "
                f"model={tuple(target_tensor.shape)}"
            )
        if target_tensor.device.type == "meta":
            raise ValueError(
                f"Cannot load non-LoRA tensor into meta-device target: {target_name}"
            )
        normalized[target_name] = source_tensor

    import torch

    with torch.no_grad():
        for target_name, source_tensor in normalized.items():
            target_tensor = available[target_name]
            target_tensor.copy_(
                source_tensor.to(
                    device=target_tensor.device,
                    dtype=target_tensor.dtype,
                )
            )
    return tuple(sorted(normalized))


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"{label} directory does not exist: {path}")


def merge_lora(
    pretrained_model_path: Path,
    lora_adapter_path: Path,
    output_dir: Path,
) -> Path:
    """Merge a local HunyuanOCR LoRA adapter and save a standalone model."""
    _require_directory(pretrained_model_path, "Pretrained model")
    _require_directory(lora_adapter_path, "LoRA adapter")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    print(f"Loading processor: {pretrained_model_path}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(
        pretrained_model_path,
        trust_remote_code=True,
        local_files_only=True,
        backend="pil",
    )

    print(f"Loading pretrained model: {pretrained_model_path}", file=sys.stderr)
    base_model = HunYuanVLForConditionalGeneration.from_pretrained(
        pretrained_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"Loading LoRA adapter: {lora_adapter_path}", file=sys.stderr)
    peft_config = PeftConfig.from_pretrained(
        lora_adapter_path,
        local_files_only=True,
    )
    peft_config.target_modules = _select_target_modules(
        base_model,
        peft_config.target_modules,
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        is_trainable=False,
        local_files_only=True,
        config=peft_config,
        key_mapping=_ADAPTER_KEY_MAPPING,
    )

    print("Merging LoRA weights into the pretrained model...", file=sys.stderr)
    merged_model = peft_model.merge_and_unload(safe_merge=True)

    loaded_non_lora = _load_non_lora_trainables(
        merged_model,
        lora_adapter_path,
    )
    if loaded_non_lora:
        print(
            f"Loaded {len(loaded_non_lora)} non-LoRA tensors.",
            file=sys.stderr,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model: {output_dir}", file=sys.stderr)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    _preserve_model_config(pretrained_model_path, output_dir)
    processor.save_pretrained(output_dir)
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
