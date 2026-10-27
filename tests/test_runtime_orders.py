from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gemma_backend import GemmaDecisions


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [100, 101] + ([ord(text[-1])] if text.startswith("prompt") and text[-1].isupper() else [])

    def __call__(self, prompts, **kwargs):
        return {"input_ids": [self.encode(text) for text in prompts]}


class RuntimeOrderContract(unittest.TestCase):
    def engine(self):
        engine = GemmaDecisions.__new__(GemmaDecisions)
        engine.config = {"max_questions": 16, "max_input_tokens": 4096, "max_total_tokens": 8192,
                         "batch_size": 4, "temperature": 1, "served_model": "test",
                         "model_id": "official-test", "revision": "fixed-test", "option_orders": 2,
                         "ensemble_method": "logit_mean", "record_order_components": True}
        engine.tokenizer = FakeTokenizer()
        engine.tokens = list(range(65, 91))
        engine.torch = SimpleNamespace(inference_mode=nullcontext,
                                       cuda=SimpleNamespace(synchronize=lambda: None))
        engine._score_batch = Mock(return_value=[[0, 1, 4] + [0] * 23, [4, 1, 0] + [0] * 23])
        return engine

    def test_two_equivalent_orders_share_one_batch_and_restore_the_score_scale(self):
        engine = self.engine()
        body = {"state": "example", "questions": {"rating": {"type": "score", "criteria": ["Low", "Middle", "High"]}}}
        with patch("gemma_backend.make_prompt", return_value="prompt"):
            response = engine.decide(body)
        self.assertEqual(len(engine._score_batch.call_args.args[0]), 2)
        self.assertEqual(response["usage"]["input_tokens"], 4)
        self.assertGreater(response["answers"]["rating"]["score"], 1.8)
        self.assertEqual(response["metadata"]["question_orders"]["rating"], 2)
        self.assertEqual(len(response["metadata"]["order_components"]["rating"]), 2)
        self.assertEqual(response["metadata"]["generated_tokens"], 0)

    def test_effective_token_budget_is_checked_before_either_forward(self):
        engine = self.engine()
        engine.config["max_total_tokens"] = 3
        body = {"state": "example", "questions": {"rating": {"type": "score", "criteria": ["Low", "Middle", "High"]}}}
        with patch("gemma_backend.make_prompt", return_value="prompt"), self.assertRaises(ValueError):
            engine.decide(body)
        engine._score_batch.assert_not_called()
