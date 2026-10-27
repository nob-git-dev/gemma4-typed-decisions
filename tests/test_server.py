import unittest
from unittest.mock import Mock, patch

try:
    from fastapi.testclient import TestClient
    from server import app, CONFIG, gate
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "API check runs in the pinned container")
class APIContract(unittest.TestCase):
    def setUp(self):
        class Engine:
            def decide(self, body):
                return {"model": CONFIG["served_model"],
                        "answers": {k: {"type": "noul", "noul": 0.75} for k in body["questions"]}}
        app.state.engine = Engine()
        # No lifespan here: this contract test must not load weights or use a GPU.
        self.client = TestClient(app)
        self.body = {"state": "example", "questions": {"ok": {"type": "noul", "instructions": "True?"}}}

    def test_valid_typed_request(self):
        response = self.client.post("/v1/systemone", json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answers"]["ok"]["noul"], 0.75)

    def test_wrong_model_and_invalid_input_return_422(self):
        for body in [{**self.body, "model": "jev-latest"}, {**self.body, "questions": {}}]:
            self.assertEqual(self.client.post("/v1/systemone", json=body).status_code, 422)

    def test_large_body_returns_413_before_inference(self):
        response = self.client.post("/v1/systemone", content=b"x" * 131073,
                                    headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 413)

    def test_busy_inference_returns_429(self):
        gate.acquire()
        try:
            response = self.client.post("/v1/systemone", json=self.body)
            self.assertEqual(response.status_code, 429)
        finally:
            gate.release()

    def test_selected_pooling_profile_uses_its_own_backend_and_stops_the_watchdog(self):
        stopped = Mock()
        with patch.dict(CONFIG, {"backend": "vllm"}), \
                patch("server.check"), patch("server.start_watchdog", return_value=stopped), \
                patch("vllm_backend.VllmDecisions") as factory:
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
            factory.assert_called_once_with(CONFIG)
            stopped.set.assert_called_once()

    def test_unknown_backend_fails_startup(self):
        with patch.dict(CONFIG, {"backend": "unknown"}), \
                patch("server.check"), patch("server.start_watchdog"):
            with self.assertRaises(ValueError), TestClient(app):
                pass
