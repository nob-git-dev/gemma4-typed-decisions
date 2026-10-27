import math
import unittest

from decision_core import compile_question, probabilities, render_answer
from order_ensemble import combine_orders, question_variants


class OptionOrderContract(unittest.TestCase):
    def test_reversal_preserves_names_and_their_criteria(self):
        question = compile_question({"type": "choice", "instructions": "Select the matching item",
                                     "criteria": {"a": "Alpha", "b": "Beta", "c": "Gamma"}})
        variants = question_variants(question, 2)
        self.assertEqual(variants[1].names, ("c", "b", "a"))
        self.assertEqual(variants[1].descriptions, ("Gamma", "Beta", "Alpha"))
        self.assertEqual(variants[1].instructions, question.instructions)

    def test_binary_questions_retain_the_frozen_one_pass_policy(self):
        question = compile_question({"type": "noul", "instructions": "True?"})
        self.assertEqual(question_variants(question, 2), (question,))

    def test_reversed_logits_are_aligned_before_the_average(self):
        question = compile_question({"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}})
        variants = question_variants(question, 2)
        actual = combine_orders(question, variants, [[4, 1, 0], [3, 2, 1]], 1, "logit_mean")
        expected = probabilities([2.5, 1.5, 1.5])
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b)

    def test_probability_mean_is_a_distinct_predeclared_development_candidate(self):
        question = compile_question({"type": "choice", "criteria": {"a": "A", "b": "B"}})
        variants = question_variants(question, 2)
        actual = combine_orders(question, variants, [[4, 1], [3, 1]], 1, "probability_mean")
        left, right = probabilities([4, 1]), probabilities([1, 3])
        self.assertAlmostEqual(actual[0], (left[0] + right[0]) / 2)
        self.assertAlmostEqual(sum(actual), 1)

    def test_score_positions_are_restored_before_the_expected_score(self):
        question = compile_question({"type": "score", "criteria": ["Low", "Middle", "High"]})
        variants = question_variants(question, 2)
        probs = combine_orders(question, variants, [[0, 1, 4], [4, 1, 0]], 1, "logit_mean")
        self.assertGreater(render_answer(question, probs)["score"], 1.8)

    def test_one_order_retains_the_existing_distribution(self):
        question = compile_question({"type": "choice", "criteria": {"a": "A", "b": "B"}})
        self.assertEqual(combine_orders(question, (question,), [[1, 2]], 1, "logit_mean"),
                         probabilities([1, 2]))

    def test_missing_rows_and_unknown_methods_fail(self):
        question = compile_question({"type": "choice", "criteria": {"a": "A", "b": "B"}})
        for rows, method in [([], "logit_mean"), ([[1, 2]], "unknown"), ([[math.nan, 2]], "logit_mean")]:
            with self.assertRaises(ValueError):
                combine_orders(question, (question,), rows, 1, method)
