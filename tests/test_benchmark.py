import json
import http.server
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchmark import OFFICIAL_JEV, close_connections, load_cases, post, run


class BenchmarkContract(unittest.TestCase):
    def test_warm_requests_reuse_the_http_connection(self):
        ports = []

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                ports.append(self.client_address[1])
                payload = b'{"model":"test"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = "http://127.0.0.1:" + str(server.server_port) + "/test"
            post(url, {})
            post(url, {})
            self.assertEqual(len(ports), 2)
            self.assertEqual(ports[0], ports[1])
        finally:
            close_connections()
            server.shutdown()
            server.server_close()

    def test_api_key_cannot_be_sent_to_another_host(self):
        with self.assertRaises(ValueError):
            post("https://example.com/v1/systemone", {}, "secret-test-placeholder")

    def test_a_reflected_credential_cannot_be_returned_or_saved(self):
        connection = Mock()
        response = connection.getresponse.return_value
        response.status = 200
        response.read.return_value = b'{"echo":"secret-test-placeholder"}'
        try:
            with patch("benchmark.http.client.HTTPSConnection", return_value=connection):
                with self.assertRaises(ValueError):
                    post(OFFICIAL_JEV, {}, "secret-test-placeholder")
        finally:
            close_connections()

    def test_invalid_response_is_preserved_for_diagnosis(self):
        config = json.loads((Path(__file__).parents[1] / "config.json").read_text())
        bad = {"model": config["served_model"], "answers": {"decision": {"type": "noul", "noul": 2}}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            args = SimpleNamespace(cases=Path(__file__).parents[1] / "data/smoke.jsonl",
                                   local_only=True, warmup=0, timeout=30,
                                   output=output, local_url="http://127.0.0.1:30000/v1/systemone")
            with patch("benchmark.post", return_value=(bad, 100)):
                with self.assertRaises(ValueError):
                    run(args)
            saved = json.loads((output / "failed-response.json").read_text())
            self.assertEqual(saved["response"], bad)
            self.assertEqual(saved["backend"], "candidate")

    def test_only_the_official_baseline_may_use_rounded_scores(self):
        config = json.loads((Path(__file__).parents[1] / "config.json").read_text())
        case = {"id": "rounded-score", "state": "A mixed review", "question": {
            "type": "score", "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]},
            "expected": 1, "split": "test", "source": "synthetic"}

        def response(url, body, key, timeout):
            score = 1.79 if body["model"] == config["jev_model"] else 1.78
            answer = {"type": "score", "score": score,
                      "probabilities": {"0": 0.15, "1": 0.44, "2": 0.04, "3": 0.22, "4": 0.15}}
            return {"model": body["model"], "answers": {"decision": answer}}, 100

        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / "cases.jsonl"
            cases.write_text(json.dumps(case) + "\n")
            args = SimpleNamespace(cases=cases, local_only=False, warmup=0, timeout=30,
                                   output=Path(tmp) / "out", local_url="http://127.0.0.1:30000/v1/systemone")
            with patch.dict("os.environ", {"TYPESAFE_API_KEY": "secret-test-placeholder"}), \
                    patch("benchmark.post", side_effect=response), patch("builtins.print"):
                self.assertEqual(run(args)["verdict"], "UNVERIFIED")

    def test_explicit_candidate_profile_controls_requests_and_manifest(self):
        config = json.loads((Path(__file__).parents[1] / "config.json").read_text())
        config["served_model"] = "independent-profile-test"
        case = {"id": "profile", "state": "example", "question": {
            "type": "noul", "instructions": "Is this an example?"},
            "expected": True, "split": "test", "source": "synthetic"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "profile.json").write_text(json.dumps(config))
            (root / "cases.jsonl").write_text(json.dumps(case) + "\n")
            args = SimpleNamespace(config=root / "profile.json", cases=root / "cases.jsonl",
                                   output=root / "out", local_only=True, warmup=0, timeout=30,
                                   local_url="http://127.0.0.1:30000/v1/systemone")
            response = {"model": config["served_model"],
                        "answers": {"decision": {"type": "noul", "noul": 1.0}}}
            with patch("benchmark.post", return_value=(response, 100)) as endpoint, patch("builtins.print"):
                self.assertEqual(run(args)["candidate"]["accuracy"], 1)
            self.assertEqual(endpoint.call_args.args[1]["model"], config["served_model"])
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["config"]["served_model"], config["served_model"])

    def test_duplicate_test_inputs_cannot_inflate_sample_count(self):
        case = {"id": "a", "state": "text", "question": {"type": "noul", "instructions": "True?"},
                "expected": True, "source": "synthetic", "split": "test"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(json.dumps(case) + "\n" + json.dumps({**case, "id": "b"}) + "\n")
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_development_examples_need_an_explicit_split_and_cannot_be_heldout(self):
        case = {"id": "dev", "state": "text", "question": {"type": "noul", "instructions": "True?"},
                "expected": True, "source": "public", "split": "development"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "development.jsonl"
            path.write_text(json.dumps(case) + "\n")
            with self.assertRaises(ValueError):
                load_cases(path)
            self.assertEqual(load_cases(path, expected_split="development")[0][0]["split"], "development")

    def test_fixture_is_explicitly_synthetic_and_balanced_by_type(self):
        cases, digest = load_cases(Path(__file__).parents[1] / "data/smoke.jsonl")
        self.assertEqual(len(cases), 24)
        self.assertEqual(len(digest), 64)
        self.assertEqual({c["source"] for c in cases}, {"synthetic"})
        self.assertEqual({c["question"]["type"] for c in cases}, {"noul", "choice", "score"})

    def test_missing_baseline_credentials_fail_before_any_request_or_output(self):
        class Args:
            cases = Path(__file__).parents[1] / "data/smoke.jsonl"
            local_only = False
        with patch.dict("os.environ", {}, clear=True), patch("benchmark.post") as remote:
            with self.assertRaises(ValueError):
                run(Args())
            remote.assert_not_called()
