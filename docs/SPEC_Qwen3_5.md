# Spec: Merge Qwen3.5-0.8B DPO Adapter

## Objective

Create a local-only script that merges the final Qwen3.5-0.8B DPO LoRA adapter
into its base model and saves a standalone model for inference. The DPO adapter
was trained from the SFT adapter, so it carries both stages.

## Runtime environment

Use a dedicated Python 3.12 Conda environment with the pinned dependencies in
the repository's `requirements.txt`. Install with `uv` so the PyTorch/CUDA
wheel matches the NVIDIA driver:

```bash
conda create -n qwen35-infer python=3.12 -y
conda activate qwen35-infer
python -m pip install uv
uv pip install -r requirements.txt --torch-backend=auto
```

The same environment supports merging and vLLM inference. On hosts without
`nvcc`, start vLLM with `VLLM_USE_FLASHINFER_SAMPLER=0` to use the native
sampler, then set memory/context options for the target GPU.

## Merge command

```bash
python -m src.merge_Qwen3_5.merge_lora
```

## Project structure and style

- Add one standalone Python entry point in the source tree, with editable path constants.
- Load the full Qwen3.5 conditional-generation model in BF16 and local-only mode.
- Merge with PEFT `merge_and_unload(safe_merge=True)` and save safetensors.
- Save the adapter's processor, tokenizer, and chat template with the merged model.
- Reject invalid inputs and non-empty output before loading weights.

```python
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    base_model, dtype=torch.bfloat16, local_files_only=True
)
merged_model = peft_model.merge_and_unload(safe_merge=True)
```

## Verification and success criteria

- Running the entry point writes a standalone model and processor to the configured output.
- The saved directory contains model weights/config and inference tokenizer assets.
- The pretrained model and adapter remain unchanged; merging uses local files
  and does not start a serving process.

## Boundaries

- Always: use the pinned environment, BF16, safetensors, local-only loaders, and
  refuse a non-empty output directory.
- Never: overwrite the pretrained model or adapter, or place machine-specific
  paths, IP addresses, or credentials in documentation.

## Implementation plan

1. Add the model-specific merge entry point using the installed Transformers and PEFT APIs.
2. Run the merge from the configured environment and report the saved artifacts.

## Open questions

None. The model, adapter lineage, and output location are supplied by the user.
