"""Choose a predeclared policy using development examples only, before fresh test."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from benchmark import ROOT, load_cases
from decision_core import compile_question, json_text, render_answer
from metrics import _summary, observation
from order_ensemble import combine_orders


def choose_method(summaries):
    baseline = summaries["single"]["by_type"]
    eligible = []
    for method in ("logit_mean", "probability_mean"):
        candidate = summaries[method]["by_type"]
        if all(candidate[kind]["n"] >= 400
               and candidate[kind]["accuracy"] >= baseline[kind]["accuracy"] - 1e-12
               and candidate[kind]["mean_latency_ms"] <= 140
               for kind in ("choice", "score")):
            eligible.append(method)
    if not eligible:
        return "single"
    return max(eligible, key=lambda method: summaries[method]["macro_accuracy"])


def summarize(rows):
    groups = {kind: _summary([row for row in rows if row["kind"] == kind])
              for kind in ("noul", "choice", "score")}
    return {"all": _summary(rows), "by_type": groups,
            "macro_accuracy": sum(group["accuracy"] for group in groups.values()) / 3}


def main():
    cases_path = ROOT / ".cache/gemma4-jev/evaluation/development-20261028.jsonl"
    cases, data_hash = load_cases(cases_path, expected_split="development")
    paths = {"single": ROOT / "results/development-single-20261028",
             "orders": ROOT / "results/development-orders-20261028"}
    manifests, records = {}, {}
    for name, path in paths.items():
        manifests[name] = json.loads((path / "manifest.json").read_text())
        if manifests[name].get("dataset_split") != "development" or manifests[name]["dataset_sha256"] != data_hash:
            raise ValueError("Only the identical frozen development dataset may select the policy")
        values = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
        if len(values) != len(cases) or {v["id"] for v in values} != {c["id"] for c in cases}:
            raise ValueError("Incomplete development observations")
        records[name] = {value["id"]: value for value in values}
    summaries = {"single": summarize([record["observation"] for record in records["single"].values()])}
    derived = {method: [] for method in ("logit_mean", "probability_mean")}
    temperature = manifests["orders"]["config"]["temperature"]
    for case in cases:
        record = records["orders"][case["id"]]
        # Match the sorted JSON actually sent over HTTP, including Choice tie order.
        question = compile_question(json.loads(json_text(case["question"])))
        components = record["response"]["metadata"]["order_components"]["decision"]
        variants = [replace(question, names=tuple(component["names"]),
                            descriptions=tuple(question.descriptions[question.names.index(name)]
                                               for name in component["names"]))
                    for component in components]
        rows = [component["logits"] for component in components]
        for method in derived:
            probs = combine_orders(question, variants, rows, temperature, method)
            answer = render_answer(question, probs)
            if method == "logit_mean":
                measured = record["response"]["answers"]["decision"]
                left = [answer["noul"]] if question.kind == "noul" else list(answer["probabilities"].values())
                right = ([measured["noul"]] if question.kind == "noul" else
                         [measured["probabilities"][name] for name in question.names])
                if any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(left, right)):
                    raise ValueError("Stored component reconstruction disagrees with measured output")
            derived[method].append(observation(case["question"], case["expected"], answer, record["latency_ms"]))
    summaries.update({method: summarize(rows) for method, rows in derived.items()})
    selected = choose_method(summaries)
    config = dict(manifests["single" if selected == "single" else "orders"]["config"])
    config.update(option_orders=1 if selected == "single" else 2,
                  ensemble_method="logit_mean" if selected == "single" else selected,
                  record_order_components=False,
                  served_model="gemma4-26b-fp8-decisions" if selected == "single" else "gemma4-26b-fp8-order-decisions")
    config_path = ROOT / "config.final.json"
    config_text = json.dumps(config, indent=2) + "\n"
    evidence = {"selected": selected, "dataset_sha256": data_hash, "summaries": summaries,
                "rule": "Nonbinary development accuracies preserved; each mean HTTP latency <=140ms; max macro accuracy; logit tie preference",
                "probability_mean_timing": "Same measured two-order GPU computation; probabilities combined offline for development selection",
                "manifests_sha256": {name: hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                                      for name, path in paths.items()},
                "final_config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
                "fresh_test_predictions_examined": False}
    with (ROOT / "evidence/refinement-selection-20261028.json").open("x") as target:
        json.dump(evidence, target, indent=2)
        target.write("\n")
    with config_path.open("x") as target:
        target.write(config_text)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
