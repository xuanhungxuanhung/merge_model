from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.merge_model_LoRA import merge_lora


class MergeLoraTests(unittest.TestCase):
    def test_main_uses_configured_output_path(self) -> None:
        pretrained_path = Path("/models/pretrained")
        adapter_path = Path("/models/adapter")
        output_path = Path("/models/merged")

        with (
            patch.object(
                merge_lora,
                "PRETRAINED_MODEL_PATH",
                pretrained_path,
            ),
            patch.object(
                merge_lora,
                "LORA_ADAPTER_PATH",
                adapter_path,
            ),
            patch.object(
                merge_lora,
                "OUTPUT_MODEL_PATH",
                output_path,
            ),
            patch.object(merge_lora, "merge_lora") as merge,
        ):
            exit_code = merge_lora.main()

        self.assertEqual(exit_code, 0)
        merge.assert_called_once_with(
            pretrained_path,
            adapter_path,
            output_path,
        )

    def test_merges_local_adapter_and_saves_standalone_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pretrained_path = root / "pretrained"
            adapter_path = root / "adapter"
            output_path = root / "merged"
            pretrained_path.mkdir()
            adapter_path.mkdir()
            source_config = '{"rope_scaling":{"type":"xdrope"}}\n'
            (pretrained_path / "config.json").write_text(source_config)

            processor = Mock()
            base_model = Mock()
            base_model.named_modules.return_value = [("model.layers.0.self_attn.q_proj", Mock())]
            peft_config = Mock(
                target_modules={"model.layers.0.self_attn.q_proj"}
            )
            peft_model = Mock()
            merged_model = Mock()
            peft_model.merge_and_unload.return_value = merged_model

            with (
                patch(
                    "transformers.AutoProcessor.from_pretrained",
                    return_value=processor,
                ) as load_processor,
                patch(
                    "transformers.HunYuanVLForConditionalGeneration.from_pretrained",
                    return_value=base_model,
                ) as load_base_model,
                patch(
                    "peft.PeftConfig.from_pretrained",
                    return_value=peft_config,
                ) as load_adapter_config,
                patch(
                    "peft.PeftModel.from_pretrained",
                    return_value=peft_model,
                ) as load_adapter,
            ):
                result = merge_lora.merge_lora(
                    pretrained_path,
                    adapter_path,
                    output_path,
                )
                saved_config = (output_path / "config.json").read_text()

        self.assertEqual(result, output_path)
        self.assertTrue(load_processor.call_args.kwargs["local_files_only"])
        self.assertEqual(load_processor.call_args.kwargs["backend"], "pil")
        self.assertNotIn("use_fast", load_processor.call_args.kwargs)
        self.assertTrue(load_base_model.call_args.kwargs["local_files_only"])
        self.assertTrue(load_adapter_config.call_args.kwargs["local_files_only"])
        self.assertTrue(load_adapter.call_args.kwargs["local_files_only"])
        self.assertIs(load_adapter.call_args.args[0], base_model)
        self.assertIs(load_adapter.call_args.kwargs["config"], peft_config)
        peft_model.merge_and_unload.assert_called_once_with(safe_merge=True)
        merged_model.save_pretrained.assert_called_once_with(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        processor.save_pretrained.assert_called_once_with(output_path)
        self.assertEqual(json.loads(saved_config)["rope_parameters"]["rope_type"], "xdrope")

    def test_rejects_non_empty_output_before_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pretrained_path = root / "pretrained"
            adapter_path = root / "adapter"
            output_path = root / "merged"
            pretrained_path.mkdir()
            adapter_path.mkdir()
            output_path.mkdir()
            (output_path / "old-file").touch()

            with (
                patch(
                    "transformers.HunYuanVLForConditionalGeneration.from_pretrained"
                ) as load_base_model,
                self.assertRaisesRegex(FileExistsError, "not empty"),
            ):
                merge_lora.merge_lora(
                    pretrained_path,
                    adapter_path,
                    output_path,
                )

        load_base_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
