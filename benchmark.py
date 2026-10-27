"""An independently written paired HTTP benchmark; no Jev reproduction code."""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import time
import urllib.error
import urllib.parse

from decision_core import compile_question, json_text
from metrics import _summary, compare, observation

OFFICIAL_JEV = "https://api.typesafe.ai/v1/systemone"
ROOT = Path(__file__).resolve().parent


CONNECTIONS = {}


def close_connections():
    for connection in CONNECTIONS.values():
        connection.close()
    CONNECTIONS.clear()


def post(url, body, key=None, timeout=30):
    headers = {"Content-Type": "application/json"}
    if key:
        if url != OFFICIAL_JEV:
            raise ValueError("Credentials may only be sent to the official Jev endpoint")
        headers["Authorization"] = "Bearer " + key
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Expected an HTTP(S) endpoint")
    identity = (parsed.scheme, parsed.hostname, parsed.port, timeout)
    if identity not in CONNECTIONS:
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        CONNECTIONS[identity] = cls(parsed.hostname, parsed.port, timeout=timeout)
    connection = CONNECTIONS[identity]
    payload = json_text(body).encode("utf-8")
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    started = time.perf_counter()
    try:
        connection.request("POST", target, body=payload, headers=headers)
        response = connection.getresponse()
        raw = response.read(4 * 1024**2 + 1)
        if not 200 <= response.status < 300:
            raise urllib.error.HTTPError(url, response.status, response.reason,
                                         response.headers, None)
    except Exception:
        connection.close()
        CONNECTIONS.pop(identity, None)
        raise
    elapsed = (time.perf_counter() - started) * 1000
    if len(raw) > 4 * 1024**2:
        raise ValueError("Response exceeds size limit")
    if key and key.encode("utf-8") in raw:
        connection.close()
        CONNECTIONS.pop(identity, None)
        raise ValueError("Credential reflected by endpoint; response was discarded")
    return json.loads(raw), elapsed


def load_cases(path, expected_split="test"):
    raw = path.read_bytes()
    cases = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = set()
    for case in cases:
        if not isinstance(case["id"], str) or case["id"] in ids:
            raise ValueError("Dataset case identifiers must be unique strings")
        ids.add(case["id"])
        compile_question(case["question"])
        if case.get("split") != expected_split or case.get("source") not in ("synthetic", "human", "public"):
            raise ValueError("Every case needs the declared split and explicit label provenance")
        kind, expected = case["question"]["type"], case["expected"]
        if kind == "noul" and type(expected) is not bool:
            raise ValueError("Noul label must be boolean")
        if kind == "choice" and expected not in case["question"]["criteria"]:
            raise ValueError("Choice label must be in the rubric")
        if kind == "score" and (type(expected) is not int or not 0 <= expected < len(case["question"]["criteria"])):
            raise ValueError("Score label must be a rubric index")
    if not cases:
        raise ValueError("Empty dataset")
    # Exact duplicate inputs cannot increase the independent sample count.
    inputs = [json_text({"state": c["state"], "question": c["question"]}) for c in cases]
    if len(set(inputs)) != len(inputs):
        raise ValueError("Duplicate decision inputs do not count as independent examples")
    return cases, hashlib.sha256(raw).hexdigest()


def request_for(case, model):
    return {"model": model, "state": case["state"],
            "questions": {"decision": case["question"]}}


def run(args):
    config = json.loads(getattr(args, "config", ROOT / "config.json").read_text())
    development = getattr(args, "development", False)
    cases, digest = load_cases(args.cases, expected_split="development" if development else "test")
    key = os.environ.get("TYPESAFE_API_KEY")
    if not args.local_only and not key:
        raise ValueError("Set TYPESAFE_API_KEY for official Jev comparison; no baseline was fabricated")
    if args.warmup < 0 or args.timeout <= 0:
        raise ValueError("Invalid warmup or timeout")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    endpoints = {"candidate": (args.local_url, config["served_model"], None)}
    if not args.local_only:
        endpoints["baseline"] = (OFFICIAL_JEV, config["jev_model"], key)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.cases.name, "dataset_sha256": digest,
        "n_requests_per_backend": len(cases), "warmup_per_backend": args.warmup,
        "environment": {"machine": platform.machine()},
        "config": config, "timing": "client HTTP wall time, persistent connections, no retries, warmup excluded",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in ROOT.glob("*.py")},
        "dataset_sources": sorted({c["source"] for c in cases}),
        "dataset_split": "development" if development else "test",
        "evaluated_types": sorted({c["question"]["type"] for c in cases}),
        "response_validation": {"candidate_score": "strict expectation",
                                "official_score_probability_quantum": 0.01,
                                "official_score": "audit consistency; modal accuracy and reported-score MAE are unchanged"},
    }
    (output / "manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    rows = {name: [] for name in endpoints}
    try:
        for name, (url, model, auth) in endpoints.items():
            for _ in range(args.warmup):
                response, _ = post(url, request_for(cases[0], model), auth, args.timeout)
                if response.get("model") != model:
                    raise ValueError(name + " warmup returned a different model")
        with (output / "requests.jsonl").open("x") as log:
            for index, case in enumerate(cases):
                order = list(endpoints)
                if index % 2:
                    order.reverse()
                for name in order:
                    url, model, auth = endpoints[name]
                    response, elapsed = post(url, request_for(case, model), auth, args.timeout)
                    record = {"id": case["id"], "backend": name,
                              "latency_ms": elapsed, "response": response}
                    try:
                        if response.get("model") != model:
                            raise ValueError(name + " returned a different model")
                        answers = response.get("answers", {})
                        if set(answers) != {"decision"}:
                            raise ValueError("Missing or unexpected benchmark answers")
                        row = observation(case["question"], case["expected"], answers["decision"], elapsed,
                                          probability_quantum=0.01 if name == "baseline" else None,
                                          audit_reported_score=name == "baseline")
                    except Exception as exc:
                        record["validation_error"] = {"type": type(exc).__name__, "message": str(exc)}
                        (output / "failed-response.json").write_text(
                            json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                        raise
                    rows[name].append(row)
                    record["observation"] = row
                    log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(cases):
                    print(json.dumps({"completed": index + 1, "total": len(cases)}), flush=True)
        if args.local_only:
            summary = {"verdict": "UNVERIFIED", "reason": "No Jev baseline measured",
                       "candidate": _summary(rows["candidate"])}
        else:
            evidence = ("development" if development else
                        "smoke" if any(c["source"] == "synthetic" for c in cases) else "heldout")
            summary = compare(rows["baseline"], rows["candidate"], evidence_kind=evidence)
        summary["by_type"] = {
            kind: {name: _summary([r for r in values if r["kind"] == kind])
                   for name, values in rows.items()}
            for kind in provenance["evaluated_types"]}
    except Exception as exc:
        # Keep completed measurements; an HTTP or schema failure cannot become a PASS.
        summary = {"verdict": "UNVERIFIED", "reason": type(exc).__name__,
                   "completed": {name: len(values) for name, values in rows.items()}}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/smoke.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--local-url", default="http://127.0.0.1:30000/v1/systemone")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, OSError, urllib.error.URLError) as exc:
        parser.exit(2, "Benchmark stopped: " + str(exc) + "\n")
    if result["verdict"] == "FAIL":
        raise SystemExit(1)
    if result["verdict"] != "PASS" and not args.local_only:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
