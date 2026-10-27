import unittest

from select_refinement import choose_method


class DevelopmentSelectionContract(unittest.TestCase):
    def summary(self, choice=0.85, score=0.55, latency=130, macro=None):
        return {"macro_accuracy": (0.9 + choice + score) / 3 if macro is None else macro,
                "by_type": {"choice": {"n": 400, "accuracy": choice, "mean_latency_ms": latency},
                            "score": {"n": 400, "accuracy": score, "mean_latency_ms": latency},
                            "noul": {"n": 400, "accuracy": 0.9, "mean_latency_ms": 108}}}

    def test_both_nonbinary_types_must_preserve_quality(self):
        values = {"single": self.summary(), "logit_mean": self.summary(0.95, 0.54),
                  "probability_mean": self.summary(0.86, 0.56)}
        self.assertEqual(choose_method(values), "probability_mean")

    def test_slow_or_incomplete_candidates_cannot_be_selected(self):
        values = {"single": self.summary(), "logit_mean": self.summary(0.95, 0.65, 141),
                  "probability_mean": self.summary(0.95, 0.65)}
        values["probability_mean"]["by_type"]["score"]["n"] = 399
        self.assertEqual(choose_method(values), "single")

    def test_predeclared_tie_prefers_logit_mean(self):
        values = {"single": self.summary(), "logit_mean": self.summary(0.86, 0.56),
                  "probability_mean": self.summary(0.86, 0.56)}
        self.assertEqual(choose_method(values), "logit_mean")

    def test_highest_macro_accuracy_among_eligible_candidates_wins(self):
        values = {"single": self.summary(), "logit_mean": self.summary(0.86, 0.56),
                  "probability_mean": self.summary(0.87, 0.57)}
        self.assertEqual(choose_method(values), "probability_mean")
