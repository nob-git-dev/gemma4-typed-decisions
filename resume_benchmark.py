"""Revalidate and extend an interrupted paired run without remeasuring its prefix."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

from benchmark import OFFICIAL_JEV, ROOT, close_connections, load_cases, post, request_for
from metrics import _summary, compare, observation

LOCAL_URL = "http://127.0.0.1:30000/v1/systemone"
RUNTIME_SOURCES = ("decision_core.py", "gemma_backend.py", "vllm_backend.py",
                   "bounded_worker.py", "server.py", "safety.py", "order_ensemble.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prefix(prior, config, cases, dataset_hash):
    manifest = json.loads((prior / "manifest.json").read_text())
    if manifest["dataset_sha256"] != dataset_hash or manifest["config"] != config:
        raise ValueError("Dataset or runtime configuration changed; reuse is forbidden")
    for name in RUNTIME_SOURCES:
        if manifest["source_sha256"][name] != digest(ROOT / name):
            raise ValueError("Inference policy changed; reuse is forbidden")
    snapshot = prior / "source" / "benchmark.py"
    if digest(snapshot) != manifest["source_sha256"]["benchmark.py"]:
        raise ValueError("Original measurement source snapshot is missing or different")
    original = ast.parse(snapshot.read_text())
    for function in (post, request_for):
        old = next(node for node in original.body if isinstance(node, ast.FunctionDef)
                   and node.name == function.__name__)
        current = ast.parse(inspect.getsource(function)).body[0]
        if ast.dump(old) != ast.dump(current):
            raise ValueError("HTTP timing or request construction changed")
    versions = {"candidate": config["served_model"], "baseline": config["jev_model"]}
    records = {}
    for line in (prior / "requests.jsonl").read_text().splitlines():
        record = json.loads(line)
        key = record["id"], record["backend"]
        if key in records or record["backend"] not in versions:
            raise ValueError("Duplicate or unexpected previous observations")
        if record["response"].get("model") != versions[record["backend"]]:
            raise ValueError("Previous response came from a different model")
        records[key] = record
    prefix = []
    incomplete = False
    for case in cases:
        complete = all((case["id"], backend) in records for backend in versions)
        if not complete:
            incomplete = True
            continue
        if incomplete:
            raise ValueError("Only an uninterrupted leading prefix may be reused")
        for backend in versions:
            record = dict(records[(case["id"], backend)])
            record["observation"] = observation(
                case["question"], case["expected"], record["response"]["answers"]["decision"],
                record["latency_ms"], probability_quantum=0.01 if backend == "baseline" else None,
                audit_reported_score=backend == "baseline")
            record["measurement_origin"] = record.get("measurement_origin", record.get("reused_from", str(prior.resolve())))
            record["reused_from"] = str(prior.resolve())
            prefix.append(record)
    if not prefix or len(prefix) // 2 == len(cases):
        raise ValueError("Expected a nonempty, incomplete measured prefix")
    if set(records) - {(c["id"], b) for c in cases for b in versions}:
        raise ValueError("Previous observations contain unknown case IDs")
    return prefix, manifest


def read_key(path):
    if path.stat().st_mode & 0o077:
        raise ValueError("Credential file must be private")
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) != 1 or not lines[0].startswith("TYPESAFE_API_KEY="):
        raise ValueError("Invalid credential file format; contents were not displayed")
    key = lines[0].partition("=")[2]
    if not key or any(ord(c) <= 32 or ord(c) >= 127 for c in key):
        raise ValueError("Invalid credential value; contents were not displayed")
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.vllm.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--key-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    old = json.loads((args.prior / "manifest.json").read_text())
    cases, dataset_hash = load_cases(Path(old["dataset"]))
    config = json.loads(args.config.read_text())
    prefix, old = load_prefix(args.prior, config, cases, dataset_hash)
    offset = len(prefix) // 2
    print(json.dumps({"revalidated_pairs": offset, "remaining_pairs": len(cases) - offset}), flush=True)
    if args.dry_run:
        return
    if args.output is None:
        raise ValueError("Specify a new output directory")
    key = read_key(args.key_file)
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    hashes = {}
    for path in ROOT.glob("*.py"):
        (source / path.name).write_bytes(path.read_bytes())
        hashes[path.name] = digest(path)
    provenance = {**old, "created_at": datetime.now(timezone.utc).isoformat(),
                  "source_sha256": hashes,
                  "endpoints": {"candidate": LOCAL_URL, "baseline": OFFICIAL_JEV},
                  "response_validation": {"candidate": "strict", "baseline_probability_quantum": 0.01,
                                          "baseline_score": "audit deviation; use modal accuracy and reported scalar MAE"},
                  "reuse": {"directory": str(args.prior.resolve()), "paired_prefix": offset,
                            "manifest_sha256": digest(args.prior / "manifest.json"),
                            "requests_sha256": digest(args.prior / "requests.jsonl"),
                            "http_protocol": "post and request_for ASTs match the saved original source",
                            "legacy_endpoint_basis": "observed original task command used the default loopback URL"},
                  "new_requests_per_backend": len(cases) - offset,
                  "timing_segments": "original prefix; pause; three fresh warmups; alternating continuation"}
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rows = {"candidate": [], "baseline": []}
    endpoints = {"candidate": (LOCAL_URL, config["served_model"], None),
                 "baseline": (OFFICIAL_JEV, config["jev_model"], key)}
    try:
        with (args.output / "requests.jsonl").open("x") as log:
            for record in prefix:
                rows[record["backend"]].append(record["observation"])
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            for url, model, auth in endpoints.values():
                for _ in range(3):
                    response, _ = post(url, request_for(cases[offset], model), auth, 30)
                    if response.get("model") != model:
                        raise ValueError("Warmup returned a different model")
            for index in range(offset, len(cases)):
                case = cases[index]
                order = list(endpoints) if index % 2 == 0 else list(reversed(endpoints))
                for backend in order:
                    url, model, auth = endpoints[backend]
                    response, elapsed = post(url, request_for(case, model), auth, 30)
                    record = {"id": case["id"], "backend": backend, "latency_ms": elapsed, "response": response}
                    try:
                        if response.get("model") != model or set(response.get("answers", {})) != {"decision"}:
                            raise ValueError("Unexpected measured model or answer set")
                        record["observation"] = observation(case["question"], case["expected"],
                            response["answers"]["decision"], elapsed,
                            probability_quantum=0.01 if backend == "baseline" else None,
                            audit_reported_score=backend == "baseline")
                    except Exception as exc:
                        record["validation_error"] = {"type": type(exc).__name__, "message": str(exc)}
                        (args.output / "failed-response.json").write_text(json.dumps(record, indent=2) + "\n")
                        raise
                    rows[backend].append(record["observation"])
                    log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                if (index + 1) % 25 == 0 or index + 1 == len(cases):
                    print(json.dumps({"completed": index + 1, "total": len(cases)}), flush=True)
        evidence = "smoke" if any(c["source"] == "synthetic" for c in cases) else "heldout"
        summary = compare(rows["baseline"], rows["candidate"], evidence_kind=evidence)
        summary["by_type"] = {kind: {backend: _summary([r for r in values if r["kind"] == kind])
                                     for backend, values in rows.items()}
                              for kind in sorted({c["question"]["type"] for c in cases})}
    except Exception as exc:
        summary = {"verdict": "UNVERIFIED", "reason": type(exc).__name__,
                   "completed": {name: len(values) for name, values in rows.items()}}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise
    finally:
        close_connections()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    raise SystemExit({"PASS": 0, "FAIL": 1, "UNVERIFIED": 3}[summary["verdict"]])


if __name__ == "__main__":
    main()
