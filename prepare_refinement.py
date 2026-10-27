"""Freeze separate development and fresh test data before choosing a refinement."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import urllib.parse
import urllib.request

from decision_core import json_text
from prepare_eval import SOURCES, transform


def page(source, split, offset):
    params = urllib.parse.urlencode({"dataset": source["id"], "config": source["config"],
                                     "split": split, "offset": offset, "length": 100})
    with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + params, timeout=30) as response:
        if response.headers.get("x-revision") != source["revision"]:
            raise ValueError("Source revision changed")
        data = json.load(response)
    if data.get("partial"):
        raise ValueError("Partial source response")
    return data


def freeze(mode, output):
    destination = output / (mode + "-20261028.jsonl")
    if destination.exists():
        raise ValueError("Refusing to overwrite frozen examples")
    excluded_paths = [output / "heldout.jsonl"]
    if mode == "fresh-test":
        excluded_paths.append(output / "development-20261028.jsonl")
    seen = set()
    for path in excluded_paths:
        for line in path.read_text().splitlines():
            case = json.loads(line)
            seen.add(json_text({"state": case["state"], "question": case["question"]}))
    cases, manifests = [], []
    for ordinal, original in enumerate(SOURCES):
        source = dict(original)
        if mode == "development":
            split = "train" if source["id"] == "fancyzhx/ag_news" else "validation"
            quotas = source["quotas"]
        else:
            split = "test"
            quotas = ({0: 400, 1: 200, 2: 200} if ordinal == 0 else
                      {i: 400 for i in range(4)} if ordinal == 1 else {i: 180 for i in range(5)})
        first = page(source, split, 0)
        offsets = list(range(0, first["num_rows_total"], 100))
        rng = random.Random(20261028 + ordinal)
        rng.shuffle(offsets)
        counts, scanned = Counter(), 0
        for offset in offsets:
            batch = first if offset == 0 else page(source, split, offset)
            entries = list(batch["rows"])
            rng.shuffle(entries)
            for entry in entries:
                row = entry["row"]
                label = row["label"]
                scanned += 1
                if label not in quotas or counts[label] >= quotas[label]:
                    continue
                if entry.get("truncated_cells"):
                    raise ValueError("Truncated source text")
                state, question, expected = transform(source, row)
                signature = json_text({"state": state, "question": question})
                if signature in seen:
                    continue
                seen.add(signature)
                cases.append({"id": source["id"] + ":" + split + ":" + str(entry["row_idx"]),
                              "split": "development" if mode == "development" else "test",
                              "source_split": split, "source": "public", "language": "en",
                              "state": state, "question": question, "expected": expected,
                              "source_revision": source["revision"]})
                counts[label] += 1
            if all(counts[label] == quota for label, quota in quotas.items()):
                break
        if dict(counts) != quotas:
            raise ValueError("Not enough disjoint examples for the predeclared quotas")
        record = {"dataset": source["id"], "revision": source["revision"], "source_split": split,
                  "quotas": quotas, "counts": dict(counts), "rows_scanned": scanned}
        manifests.append(record)
        print(json.dumps(record), flush=True)
    data = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
    with destination.open("x") as target:
        target.write(data)
    manifest = {"n": len(cases), "dataset_sha256": hashlib.sha256(data.encode()).hexdigest(),
                "mode": mode, "seed": 20261028, "sources": manifests,
                "excluded": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in excluded_paths},
                "selection": "Deterministically shuffled pages/rows, fixed label quotas, no model predictions used"}
    with destination.with_suffix(".sources.json").open("x") as target:
        json.dump(manifest, target, indent=2)
        target.write("\n")
    print(json.dumps({"file": str(destination), "n": len(cases), "dataset_sha256": manifest["dataset_sha256"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "fresh-test"), required=True)
    args = parser.parse_args()
    freeze(args.mode, Path("/models/evaluation"))
