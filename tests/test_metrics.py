import unittest

from metrics import compare, observation, percentile


class ComparisonContract(unittest.TestCase):
    def row(self, correct=True, latency=100.0, kind="choice"):
        return {"correct": float(correct), "latency_ms": latency, "kind": kind}

    def test_faster_candidate_and_equal_accuracy_can_pass(self):
        result = compare([self.row()] * 400, [self.row(latency=90)] * 400)
        self.assertEqual(result["verdict"], "PASS")

    def test_slow_candidate_fails_even_with_perfect_accuracy(self):
        result = compare([self.row()] * 400, [self.row(latency=110)] * 400)
        self.assertEqual(result["verdict"], "FAIL")

    def test_quality_drop_is_relative(self):
        baseline = [self.row(i < 320) for i in range(400)]
        candidate = [self.row(i < 300) for i in range(400)]
        result = compare(baseline, candidate)
        self.assertAlmostEqual(result["relative_accuracy_loss"], 0.0625)
        self.assertEqual(result["verdict"], "FAIL")

    def test_missing_small_smoke_and_zero_baseline_cannot_pass(self):
        for a, b in [([], []), ([self.row()], [self.row()]),
                     ([self.row(False)] * 400, [self.row()] * 400)]:
            self.assertNotEqual(compare(a, b)["verdict"], "PASS")
        result = compare([self.row()] * 400, [self.row()] * 400, evidence_kind="smoke")
        self.assertEqual(result["verdict"], "UNVERIFIED")
        result = compare([self.row()] * 400, [self.row()] * 400, evidence_kind="development")
        self.assertEqual(result["verdict"], "UNVERIFIED")

    def test_missing_pairs_and_per_type_sample_count(self):
        with self.assertRaises(ValueError):
            compare([self.row()], [])
        rows = [self.row(kind="noul")] * 399 + [self.row(kind="choice")] * 400
        self.assertEqual(compare(rows, rows)["verdict"], "UNVERIFIED")

    def test_unknown_and_nonfinite_response_are_rejected(self):
        q = {"type": "choice", "criteria": {"a": "A", "b": "B"}}
        for answer in [{"type": "choice", "choice": "c", "probabilities": {"a": 0.5, "b": 0.5}},
                       {"type": "choice", "choice": "a", "probabilities": {"a": float("nan"), "b": 0.5}}]:
            with self.assertRaises(ValueError):
                observation(q, "a", answer, 100)

    def test_choice_uses_the_returned_choice_when_rounded_probabilities_tie(self):
        q = {"type": "choice", "criteria": {"a": "A", "b": "B"}}
        answer = {"type": "choice", "choice": "b", "probabilities": {"a": 0.5, "b": 0.5}}
        self.assertEqual(observation(q, "b", answer, 100)["correct"], 1)

    def test_official_score_rounding_is_opt_in_and_keeps_reported_mae(self):
        q = {"type": "score", "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]}
        answer = {"type": "score", "score": 1.79,
                  "probabilities": {"0": 0.15, "1": 0.44, "2": 0.04, "3": 0.22, "4": 0.15}}
        with self.assertRaises(ValueError):
            observation(q, 1, answer, 100)
        row = observation(q, 1, answer, 100, probability_quantum=0.01)
        self.assertEqual(row["correct"], 1)
        self.assertAlmostEqual(row["normalized_mae"], 0.79 / 4)
        self.assertAlmostEqual(row["score_expectation_residual"], 0.01)

    def test_rounding_never_accepts_an_unexplained_or_out_of_range_score(self):
        q = {"type": "score", "criteria": ["Low", "High"]}
        for score in (-0.001, 1.001, 0.6):
            answer = {"type": "score", "score": score, "probabilities": {"0": 0.5, "1": 0.5}}
            with self.assertRaises(ValueError):
                observation(q, 1, answer, 100, probability_quantum=0.01)

    def test_higher_precision_numbers_cannot_claim_two_decimal_rounding(self):
        q = {"type": "score", "criteria": ["Low", "High"]}
        answer = {"type": "score", "score": 0.51, "probabilities": {"0": 0.5001, "1": 0.4999}}
        with self.assertRaises(ValueError):
            observation(q, 1, answer, 100, probability_quantum=0.01)

    def test_rounded_score_bound_respects_normalized_probability_mass(self):
        q = {"type": "score", "criteria": ["Low", "Middle", "High"]}
        answer = {"type": "score", "score": 1.02,
                  "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25}}
        with self.assertRaises(ValueError):
                observation(q, 1, answer, 100, probability_quantum=0.01)

    def test_observed_official_score_with_mass_099_is_feasible_only_with_rounding(self):
        q = {"type": "score", "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]}
        answer = {"type": "score", "score": 0.14,
                  "probabilities": {"0": 0.93, "1": 0.04, "2": 0.0, "3": 0.01, "4": 0.01}}
        with self.assertRaises(ValueError):
            observation(q, 0, answer, 100)
        row = observation(q, 0, answer, 100, probability_quantum=0.01)
        self.assertAlmostEqual(row["probability_mass_residual"], -0.01)
        self.assertAlmostEqual(row["normalized_mae"], 0.14 / 4)
        with self.assertRaises(ValueError):
            observation(q, 0, {**answer, "score": 0.09}, 100, probability_quantum=0.01)
        impossible_mean = {"type": "score", "score": 3.92,
                           "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.98}}
        with self.assertRaises(ValueError):
            observation(q, 4, impossible_mean, 100, probability_quantum=0.01)

    def test_rounded_choice_mass_is_bounded_and_reported_choice_is_preserved(self):
        q = {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}}
        answer = {"type": "choice", "choice": "c", "probabilities": {"a": 0.33, "b": 0.33, "c": 0.33}}
        self.assertEqual(observation(q, "c", answer, 100, probability_quantum=0.01)["correct"], 1)
        with self.assertRaises(ValueError):
            observation(q, "c", {**answer, "probabilities": {"a": 0.3, "b": 0.3, "c": 0.3}},
                        100, probability_quantum=0.01)

    def test_official_inconsistent_scalar_is_audited_without_inventing_accuracy_or_mae(self):
        q = {"type": "score", "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]}
        answer = {"type": "score", "score": 0.3,
                  "probabilities": {"0": 0.74, "1": 0.26, "2": 0.0, "3": 0.0, "4": 0.0}}
        with self.assertRaises(ValueError):
            observation(q, 0, answer, 100, probability_quantum=0.01)
        row = observation(q, 0, answer, 100, probability_quantum=0.01, audit_reported_score=True)
        self.assertTrue(row["reported_score_contract_deviation"])
        self.assertEqual(row["correct"], 1)
        self.assertAlmostEqual(row["normalized_mae"], 0.3 / 4)
        self.assertAlmostEqual(row["score_expectation_residual"], 0.04)
        with self.assertRaises(ValueError):
            observation(q, 0, {**answer, "score": float("nan")}, 100,
                        probability_quantum=0.01, audit_reported_score=True)

    def test_percentile_uses_interpolation(self):
        self.assertAlmostEqual(percentile([0, 100], 0.95), 95)


if __name__ == "__main__":
    unittest.main()
