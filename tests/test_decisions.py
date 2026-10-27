import math
import unittest

from decision_core import compile_question, probabilities, render_answer, validate_request


class DecisionContract(unittest.TestCase):
    def test_noul_is_probability_of_true(self):
        q = compile_question({"type": "noul", "instructions": "Is the claim true?"})
        result = render_answer(q, [0.25, 0.75])
        self.assertEqual(result, {"type": "noul", "noul": 0.75})

    def test_choice_preserves_arbitrary_names(self):
        q = compile_question({"type": "choice", "criteria": {"日本語": "Japanese", "en": "English"}})
        result = render_answer(q, [0.9, 0.1])
        self.assertEqual(result["choice"], "日本語")
        self.assertEqual(result["probabilities"], {"日本語": 0.9, "en": 0.1})

    def test_score_is_expected_value_and_preserves_rubric(self):
        q = compile_question({"type": "score", "criteria": ["low", "medium", "high"]})
        result = render_answer(q, [0.1, 0.2, 0.7])
        self.assertAlmostEqual(result["score"], 1.6)
        self.assertEqual(result["legend"], {"0": "low", "1": "medium", "2": "high"})

    def test_stable_probability_and_temperature(self):
        p = probabilities([10000.0, 10001.0])
        self.assertAlmostEqual(sum(p), 1)
        self.assertAlmostEqual(p[1], 1 / (1 + math.exp(-1)))
        self.assertLess(probabilities([0, 1], 2)[1], probabilities([0, 1], 1)[1])

    def test_rejects_nonfinite_and_invalid_temperature(self):
        for values, temperature in [([math.nan, 1], 1), ([0, math.inf], 1), ([1], 0)]:
            with self.assertRaises(ValueError):
                probabilities(values, temperature)

    def test_invalid_question_fails_without_model_work(self):
        for q in [{"type": "box"}, {"type": "noul"},
                  {"type": "choice", "criteria": {}},
                  {"type": "score", "criteria": ["x"]},
                  {"type": "score", "criteria": list(range(11))}]:
            with self.subTest(q=q), self.assertRaises(ValueError):
                compile_question(q)

    def test_request_limits_and_finite_json(self):
        valid = {"state": {"message": "text"}, "questions": {"q": {"type": "noul", "instructions": "True?"}}}
        self.assertEqual(len(validate_request(valid, 16)), 1)
        with self.assertRaises(ValueError):
            validate_request({**valid, "state": math.nan}, 16)
        with self.assertRaises(ValueError):
            validate_request({**valid, "questions": {}}, 16)
        with self.assertRaises(ValueError):
            validate_request(valid, 0)


if __name__ == "__main__":
    unittest.main()
