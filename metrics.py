"""Paired evaluation. Local/remote agreement is not ground-truth accuracy."""
import math
import random
from collections import Counter


def percentile(values, fraction):
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Nonempty values and percentile in [0,1] required")
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _rounded_probability_bounds(values, quantum):
    if (not isinstance(quantum, (int, float)) or isinstance(quantum, bool)
            or not math.isfinite(quantum) or not 0 < quantum <= 0.01):
        raise ValueError("Invalid response rounding quantum")
    if any(abs(value - round(value / quantum) * quantum) > 1e-8
           for value in values):
        raise ValueError("Response does not use the specified rounding precision")
    lower = [max(0, value - quantum / 2) for value in values]
    upper = [min(1, value + quantum / 2) for value in values]
    if sum(lower) > 1 + 1e-8 or sum(upper) < 1 - 1e-8:
        raise ValueError("Rounded probabilities cannot form a normalized distribution")
    return lower, upper


def _rounded_score_interval(values, score, quantum):
    """Bound the mean of an unrounded, normalized distribution, without changing it."""
    lower, upper = _rounded_probability_bounds(values, quantum)
    if abs(score - round(score / quantum) * quantum) > 1e-8:
        raise ValueError("Score does not use the specified rounding precision")

    def endpoint(order):
        mass = 1 - sum(lower)
        mean = sum(i * p for i, p in enumerate(lower))
        for i in order:
            added = min(mass, upper[i] - lower[i])
            mean += i * added
            mass -= added
        if abs(mass) > 1e-8:
            raise ValueError("Rounded probabilities cannot form a normalized distribution")
        return mean

    indices = range(len(values))
    # The independently reported score is rounded to the same decimal grid.
    return endpoint(indices) - quantum / 2, endpoint(reversed(indices)) + quantum / 2


def observation(question, expected, answer, latency_ms, *, probability_quantum=None,
                audit_reported_score=False):
    kind = question["type"]
    if answer.get("type") != kind or not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError("Invalid answer type or measured latency")
    row = {"kind": kind, "latency_ms": latency_ms}
    if kind == "noul":
        if not isinstance(expected, bool):
            raise ValueError("Noul ground truth must be boolean")
        p = answer.get("noul")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid noul probability")
        row.update(correct=float((p >= 0.5) == expected), brier=(p - int(expected))**2)
        return row
    if kind == "choice":
        names = list(question["criteria"])
        if expected not in names:
            raise ValueError("Choice ground truth is not in criteria")
    elif kind == "score":
        names = list(map(str, range(len(question["criteria"]))))
        if type(expected) is not int or str(expected) not in names:
            raise ValueError("Score ground truth must be a rubric index")
    else:
        raise ValueError("Unsupported benchmark question")
    probs = answer.get("probabilities", {})
    if set(probs) != set(names) or any(
            isinstance(p, bool) or not isinstance(p, (int, float))
            or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()):
        raise ValueError("Invalid answer probabilities")
    if not math.isclose(sum(probs.values()), 1, abs_tol=1e-5):
        if probability_quantum is None:
            raise ValueError("Answer probabilities do not sum to one")
        _rounded_probability_bounds(list(probs.values()), probability_quantum)
        row["probability_mass_residual"] = sum(probs.values()) - 1
    chosen = max(names, key=probs.__getitem__)
    if kind == "choice" and (answer.get("choice") not in names
                              or probs[answer["choice"]] < probs[chosen] - 1e-6):
        raise ValueError("Choice does not match its distribution")
    prediction = answer["choice"] if kind == "choice" else chosen
    row["correct"] = float(prediction == str(expected))
    row["brier"] = sum((probs[k] - float(k == str(expected)))**2 for k in names) / len(names)
    if kind == "score":
        score = answer.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= len(names) - 1):
            raise ValueError("Invalid score")
        expectation = sum(int(k) * probs[k] for k in names)
        if not math.isclose(score, expectation, abs_tol=1e-5) or "probability_mass_residual" in row:
            if probability_quantum is None:
                raise ValueError("Score must be the expectation of its distribution")
            lower, upper = _rounded_score_interval([probs[k] for k in names], score, probability_quantum)
            if not lower - 1e-8 <= score <= upper + 1e-8:
                if not audit_reported_score:
                    raise ValueError("Score differs by more than response rounding can explain")
                # A valid distribution still defines the predeclared modal-class accuracy.
                # Keep the vendor's scalar for MAE; never replace it with our own mean.
                row["reported_score_contract_deviation"] = True
                row["reported_score_feasible_interval"] = [lower, upper]
            row["score_expectation_residual"] = score - expectation
        row["normalized_mae"] = abs(score - expected) / (len(names) - 1)
    return row


def _summary(rows):
    result = {"n": len(rows)}
    if not rows:
        return result
    result.update(accuracy=sum(r["correct"] for r in rows) / len(rows),
                  mean_latency_ms=sum(r["latency_ms"] for r in rows) / len(rows),
                  p95_latency_ms=percentile([r["latency_ms"] for r in rows], 0.95))
    for key in ("brier", "normalized_mae"):
        values = [r[key] for r in rows if key in r]
        if values:
            result[key] = sum(values) / len(values)
    result["reported_score_contract_deviations"] = sum(
        bool(row.get("reported_score_contract_deviation")) for row in rows)
    return result


def _ratios(baseline, candidate):
    a, b = _summary(baseline), _summary(candidate)
    if not a.get("accuracy"):
        return None
    return {"relative_accuracy_loss": 1 - b["accuracy"] / a["accuracy"],
            "mean_latency_ratio": b["mean_latency_ms"] / a["mean_latency_ms"],
            "p95_latency_ratio": b["p95_latency_ms"] / a["p95_latency_ms"]}


def _paired_intervals(baseline, candidate, repeats=500):
    rng = random.Random(20261027)
    loss, latency = [], []
    for _ in range(repeats):
        indices = rng.choices(range(len(baseline)), k=len(baseline))
        ratios = _ratios([baseline[i] for i in indices], [candidate[i] for i in indices])
        if ratios is None:
            return {}
        loss.append(ratios["relative_accuracy_loss"])
        latency.append(ratios["mean_latency_ratio"])
    return {"relative_accuracy_loss_upper95": percentile(loss, 0.95),
            "mean_latency_ratio_upper95": percentile(latency, 0.95)}


def compare(baseline, candidate, evidence_kind="heldout", minimum_per_type=400):
    if len(baseline) != len(candidate):
        raise ValueError("Paired observations must have the same length")
    for a, b in zip(baseline, candidate):
        if a["kind"] != b["kind"]:
            raise ValueError("Paired question types differ")
        for row in (a, b):
            if row["correct"] not in (0, 1) or not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0:
                raise ValueError("Invalid observations")
    result = {"verdict": "UNVERIFIED", "baseline": _summary(baseline),
              "candidate": _summary(candidate), "evidence_kind": evidence_kind,
              "limits": {"relative_accuracy_loss": 0.05, "latency_ratio": 1.05,
                         "minimum_per_type": minimum_per_type}}
    if not baseline:
        result["reason"] = "No paired measured evidence"
        return result
    ratios = _ratios(baseline, candidate)
    if ratios:
        result.update(ratios)
    counts = Counter(r["kind"] for r in baseline)
    if evidence_kind != "heldout" or min(counts.values()) < minimum_per_type:
        result["reason"] = "Insufficient independent held-out examples per question type"
        return result
    groups = {"all": list(range(len(baseline)))}
    groups.update({kind: [i for i, r in enumerate(baseline) if r["kind"] == kind] for kind in counts})
    result["groups"] = {}
    failures = []
    for name, indices in groups.items():
        a, b = [baseline[i] for i in indices], [candidate[i] for i in indices]
        metrics = _ratios(a, b)
        if metrics is None:
            result["reason"] = "Zero baseline accuracy makes relative loss undefined"
            return result
        metrics.update(_paired_intervals(a, b))
        result["groups"][name] = metrics
        if (metrics["relative_accuracy_loss"] > 0.05 + 1e-12
                or metrics["mean_latency_ratio"] > 1.05 + 1e-12
                or metrics["p95_latency_ratio"] > 1.05 + 1e-12):
            failures.append(name)
    if failures:
        result.update(verdict="FAIL", reason="Point estimates exceed target: " + ", ".join(failures))
    elif any(g.get("relative_accuracy_loss_upper95", math.inf) > 0.05 + 1e-12
             or g.get("mean_latency_ratio_upper95", math.inf) > 1.05 + 1e-12
             for g in result["groups"].values()):
        result["reason"] = "95% uncertainty bound does not establish the target"
    else:
        result["verdict"] = "PASS"
    return result
