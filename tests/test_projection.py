import json
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

from gemma_backend import selected_logits
from safety import violation


@unittest.skipIf(torch is None, "torch check runs inside the pinned ARM64 container")
class ProjectionContract(unittest.TestCase):
    def test_selected_rows_match_full_head_with_softcap_and_bias(self):
        torch.manual_seed(41)
        hidden = torch.randn(4, 32)
        weight = torch.randn(128, 32)
        bias = torch.randn(128)
        indices = [6, 12, 35, 100]
        full = torch.nn.functional.linear(hidden, weight, bias)
        expected = torch.tanh(full[:, indices] / 30) * 30
        actual = selected_logits(hidden, weight, indices, 30, bias)
        torch.testing.assert_close(actual, expected)

    def test_backbone_readout_matches_the_standard_gemma_model(self):
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig
        torch.manual_seed(42)
        config = Gemma4TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            num_global_key_value_heads=1, head_dim=16, global_head_dim=16,
            max_position_embeddings=64, sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
            hidden_size_per_layer_input=0, enable_moe_block=False,
            final_logit_softcapping=30.0)
        model = Gemma4ForCausalLM(config).eval()
        ids = torch.tensor([[2, 10, 20, 30]])
        candidates = [3, 8, 12]
        with torch.inference_mode():
            expected = model(input_ids=ids, use_cache=False).logits[:, -1, candidates]
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
            actual = selected_logits(hidden, model.get_output_embeddings().weight,
                                     candidates, config.final_logit_softcapping)
        torch.testing.assert_close(actual, expected)


class ResourceContract(unittest.TestCase):
    def test_every_limit_is_independent_and_startup_is_stricter(self):
        limits = json.loads((Path(__file__).parents[1] / "config.json").read_text())
        self.assertGreaterEqual(limits["startup_available_gib"], 116)
        normal = {"available_gib": 117, "gpu_c": 45, "thermal_c": 50}
        self.assertIsNone(violation(normal, limits, True))
        for key, value in [("available_gib", 19.9), ("gpu_c", 80), ("thermal_c", 90)]:
            self.assertIsNotNone(violation({**normal, key: value}, limits))
        self.assertIsNotNone(violation({**normal, "available_gib": 115.9}, limits, True))
        self.assertIsNone(violation({**normal, "available_gib": 115.9}, limits))
