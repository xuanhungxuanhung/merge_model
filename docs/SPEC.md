# Spec: Merge HunyuanOCR 1.5 LoRA

## Objective

Provide one local-only Python script that merges a HunyuanOCR 1.5 PEFT
LoRA adapter into its pretrained base model. The merged model and processor
must be saved together so the output directory can be loaded without PEFT.

## Tech Stack

- Python 3
- PyTorch
- Transformers (`HunYuanVLForConditionalGeneration`, `AutoProcessor`)
- PEFT (`PeftModel`)

## Commands

- Run: `conda run --no-capture-output -n sft_hunyuanocr python tools/merge_model_LoRA/merge_lora.py`
- Test: `conda run --no-capture-output -n sft_hunyuanocr python -m unittest discover -s tools/merge_model_LoRA -p test_merge_lora.py`
- Compile: `python3 -m py_compile tools/merge_model_LoRA/merge_lora.py tools/merge_model_LoRA/test_merge_lora.py`

## Project Structure

- `tools/merge_model_LoRA/merge_lora.py`: executable merge script
- `tools/merge_model_LoRA/test_merge_lora.py`: isolated unit tests
- `tools/merge_model_LoRA/merged_model/`: generated output, not source

## Code Style

Use typed `pathlib.Path` values, explicit validation, and a small callable
entry point:

```python
def merge_lora(
    pretrained_model_path: Path,
    lora_adapter_path: Path,
    output_dir: Path,
) -> Path:
    ...
```

`PRETRAINED_MODEL_PATH`, `LORA_ADAPTER_PATH`, and `OUTPUT_MODEL_PATH` are
user-editable constants at the top of the script.

## Testing Strategy

Use `unittest` with mocked model loaders. Verify validation, local-only model
loading, safe LoRA merging, and saving both the model and processor without
loading the multi-gigabyte model during the test suite.

## Boundaries

- Always: validate both input directories, use local files only, merge in
  BF16, save with safetensors, and preserve processor assets.
- Ask first: adding dependencies.
- Never: download a model, overwrite a non-empty output directory, or modify
  the source pretrained/adaptor directories.

## Success Criteria

- The pretrained, adapter, and output paths are constants at the top of the
  script with local defaults from this repository.
- Running the script performs `merge_and_unload(safe_merge=True)`.
- Output is written to the configured `OUTPUT_MODEL_PATH`.
- A non-empty output directory is rejected before loading the model.
- Unit tests and Python compilation pass.

## Open Questions

None.
