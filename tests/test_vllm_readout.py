import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

try:
    import torch
except ImportError:
    torch = None

from gemma_backend import selected_logits
from vllm_backend import VllmDecisions, checkpoint_rows


@unittest.skipIf(torch is None, "Runs in the pinned ARM64 container")
class PoolingReadoutContract(unittest.TestCase):
    def test_only_the_requested_official_tied_head_rows_are_read(self):
        from safetensors.torch import save_file
        weight = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "model.language_model.embed_tokens.weight"
            save_file({name: weight}, root / "weights.safetensors")
            (root / "config.json").write_text(json.dumps({"text_config": {
                "tie_word_embeddings": True, "final_logit_softcapping": 30.0}}))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {name: "weights.safetensors"}}))
            actual, cap = checkpoint_rows(root, [2, 6])
            torch.testing.assert_close(actual, weight[[2, 6]])
            self.assertEqual(cap, 30)

    def test_an_untied_checkpoint_cannot_silently_use_input_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"text_config":{"tie_word_embeddings":false}}')
            with self.assertRaises(ValueError):
                checkpoint_rows(root, [1, 2])

    def test_raw_last_hidden_states_are_projected_without_normalization_or_generation(self):
        engine = VllmDecisions.__new__(VllmDecisions)
        engine.torch = torch
        engine.device = torch.device("cpu")
        engine.tokens = [11, 12, 13]
        engine.softcap = 30.0
        engine.readout_weight = torch.tensor([[2, 0], [0, 2], [1, 1]], dtype=torch.bfloat16)
        engine.tokenizer = Mock()
        engine.tokenizer.encode.side_effect = [[1, 2], [3, 4, 5]]
        hidden = [[3.0, 4.0], [6.0, 8.0]]
        engine.llm = Mock()
        engine.llm.embed.return_value = [SimpleNamespace(outputs=SimpleNamespace(embedding=h)) for h in hidden]
        actual = engine._score_batch(["first", "second"])
        expected = selected_logits(torch.tensor(hidden, dtype=torch.bfloat16),
                                   engine.readout_weight, [0, 1, 2], 30).tolist()
        self.assertEqual(actual, expected)
        engine.llm.embed.assert_called_once_with(
            [{"prompt_token_ids": [1, 2]}, {"prompt_token_ids": [3, 4, 5]}], use_tqdm=False)
        engine.llm.generate.assert_not_called()

    def test_missing_pooling_output_is_not_a_partial_answer(self):
        engine = VllmDecisions.__new__(VllmDecisions)
        engine.tokenizer = Mock()
        engine.tokenizer.encode.return_value = [1]
        engine.llm = Mock()
        engine.llm.embed.return_value = []
        with self.assertRaises(RuntimeError):
            engine._score_batch(["input"])
