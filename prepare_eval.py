"""Freeze balanced public test examples; never use model predictions as labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

from decision_core import json_text

SOURCES = [
    {"id": "stanfordnlp/snli", "revision": "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
     "config": "plain_text", "quotas": {0: 200, 1: 100, 2: 100}},
    {"id": "fancyzhx/ag_news", "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
     "config": "default", "quotas": {0: 100, 1: 100, 2: 100, 3: 100}},
    {"id": "SetFit/sst5", "revision": "e51bdcd8cd3a30da231967c1a249ba59361279a3",
     "config": "default", "quotas": {i: 80 for i in range(5)}},
]


def transform(source, row):
    dataset = source["id"]
    if dataset == "stanfordnlp/snli":
        return ({"premise": row["premise"], "hypothesis": row["hypothesis"]},
                {"type": "noul", "instructions":
                 "Does the premise entail the hypothesis? Answer true only if the hypothesis "
                 "must be true given the premise. Neutral or contradictory hypotheses are false."},
                row["label"] == 0)
    if dataset == "fancyzhx/ag_news":
        names = ("world", "sports", "business", "science_technology")
        return (row["text"], {"type": "choice", "instructions": "Classify the main topic of the news.",
                "criteria": {"world": "World and international affairs",
                             "sports": "Sports news", "business": "Business and finance",
                             "science_technology": "Science and technology"}}, names[row["label"]])
    labels = {"very negative": 0, "negative": 1, "neutral": 2, "positive": 3, "very positive": 4}
    expected = labels[row["label_text"]]
    if expected != row["label"]:
        raise ValueError("Unexpected SST-5 label ordering")
    return (row["text"], {"type": "score", "instructions": "Rate the sentiment of this movie review.",
                         "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]},
            expected)


def page(source, offset):
    params = urllib.parse.urlencode({"dataset": source["id"], "config": source["config"],
                                     "split": "test", "offset": offset, "length": 100})
    url = "https://datasets-server.huggingface.co/rows?" + params
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.headers.get("x-revision") != source["revision"]:
            raise ValueError("Dataset server revision differs from the frozen source")
        result = json.load(response)
    if result.get("partial"):
        raise ValueError("Partial dataset server response")
    return result


def main():
    config = json.loads(Path("config.json").read_text())
    output = Path(config["model_path"]).parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "heldout.jsonl"
    if destination.exists():
        raise ValueError("Frozen evaluation file already exists; refusing to overwrite")
    cases, manifest = [], []
    seen = set()
    for source in SOURCES:
        counts = Counter()
        offset = 0
        while any(counts[k] < limit for k, limit in source["quotas"].items()):
            batch = page(source, offset)
            if not batch["rows"]:
                raise ValueError("Dataset exhausted before reaching balanced quotas")
            for entry in batch["rows"]:
                row = entry["row"]
                label = row["label"]
                if label not in source["quotas"] or counts[label] >= source["quotas"][label]:
                    continue
                if entry.get("truncated_cells"):
                    raise ValueError("Dataset text was truncated")
                state, question, expected = transform(source, row)
                signature = json_text({"state": state, "question": question})
                if signature in seen:
                    continue
                seen.add(signature)
                cases.append({"id": source["id"] + ":" + str(entry["row_idx"]),
                              "split": "test", "source": "public", "language": "en",
                              "state": state, "question": question, "expected": expected,
                              "source_revision": source["revision"]})
                counts[label] += 1
            offset += len(batch["rows"])
            if offset >= batch["num_rows_total"]:
                break
        if dict(counts) != source["quotas"]:
            raise ValueError("Failed to collect the required source labels")
        manifest.append({**source, "source_url": "https://huggingface.co/datasets/" + source["id"],
                         "counts": dict(counts), "rows_scanned": offset})
        print(json.dumps(manifest[-1]), flush=True)
    data = "".join(json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n" for c in cases)
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    destination.write_text(data)
    (output / "sources.json").write_text(json.dumps({
        "sources": manifest, "dataset_sha256": digest, "n": len(cases),
        "scope": "English SNLI entailment, AG News topics and SST-5 sentiment only",
        "selection": "First unique rows satisfying label quotas; no model predictions examined",
        "labels": "Original public test labels; never Jev answers",
    }, indent=2) + "\n")
    print(json.dumps({"n": len(cases), "dataset_sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
