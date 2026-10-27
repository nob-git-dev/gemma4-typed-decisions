import json
from pathlib import Path
import tempfile
import unittest

from benchmark import ROOT
from resume_benchmark import RUNTIME_SOURCES, digest, load_prefix, read_key


class PrefixEvidenceContract(unittest.TestCase):
    def fixture(self, root, case_id="first"):
        config = {"served_model": "candidate-test", "jev_model": "baseline-test"}
        cases = [{"id": name, "question": {"type": "noul", "instructions": "True?"},
                  "expected": True} for name in ("first", "second")]
        (root / "source").mkdir()
        (root / "source/benchmark.py").write_bytes((ROOT / "benchmark.py").read_bytes())
        hashes = {name: digest(ROOT / name) for name in [*RUNTIME_SOURCES, "benchmark.py"]}
        manifest = {"config": config, "dataset_sha256": "dataset-test", "source_sha256": hashes}
        (root / "manifest.json").write_text(json.dumps(manifest))
        records = [{"id": case_id, "backend": backend, "latency_ms": 100,
                    "response": {"model": config[model_key], "answers": {"decision": {"type": "noul", "noul": 0.9}}}}
                   for backend, model_key in (("candidate", "served_model"), ("baseline", "jev_model"))]
        (root / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        return config, cases, manifest

    def test_prefix_is_revalidated_without_changing_its_measured_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = self.fixture(root)
            prefix, _ = load_prefix(root, config, cases, "dataset-test")
            self.assertEqual(len(prefix), 2)
            self.assertEqual(prefix[0]["latency_ms"], 100)
            self.assertEqual(prefix[0]["observation"]["correct"], 1)
            self.assertEqual(prefix[0]["reused_from"], str(root.resolve()))

    def test_changed_runtime_or_dataset_cannot_reuse_previous_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, manifest = self.fixture(root)
            for altered, data in [({**config, "served_model": "different"}, "dataset-test"),
                                  (config, "different-dataset")]:
                with self.assertRaises(ValueError):
                    load_prefix(root, altered, cases, data)
            manifest["source_sha256"]["server.py"] = "not-the-same-runtime"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_prefix(root, config, cases, "dataset-test")

    def test_a_gap_cannot_cherry_pick_later_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = self.fixture(root, case_id="second")
            with self.assertRaises(ValueError):
                load_prefix(root, config, cases, "dataset-test")

    def test_duplicate_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = self.fixture(root)
            path = root / "requests.jsonl"
            path.write_text(path.read_text() * 2)
            with self.assertRaises(ValueError):
                load_prefix(root, config, cases, "dataset-test")

    def test_key_file_permissions_and_format_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "credential"
            path.write_text("TYPESAFE_API_KEY=public-test-placeholder\n")
            path.chmod(0o600)
            self.assertEqual(read_key(path), "public-test-placeholder")
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                read_key(path)
