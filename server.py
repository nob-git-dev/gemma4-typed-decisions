"""Loopback-only shared typed inference API, without text generation."""
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
import uvicorn

from decision_core import validate_request
from gemma_backend import GemmaDecisions
from safety import check, start_watchdog

CONFIG = json.loads(Path(os.environ.get("GEMMA4_JEV_CONFIG",
                                       Path(__file__).with_name("config.json"))).read_text())
gate = threading.BoundedSemaphore(1)


@asynccontextmanager
async def lifespan(app):
    check(CONFIG, startup=True)
    stopped = start_watchdog(CONFIG)
    try:
        if CONFIG.get("backend", "transformers") == "transformers":
            app.state.engine = GemmaDecisions(CONFIG)
        elif CONFIG["backend"] == "vllm":
            from vllm_backend import VllmDecisions
            app.state.engine = VllmDecisions(CONFIG)
        else:
            raise ValueError("Unsupported inference backend")
        check(CONFIG)
        yield
    finally:
        stopped.set()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ready", "model": CONFIG["served_model"]}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": CONFIG["served_model"], "object": "model",
                                       "capabilities": ["noul", "choice", "score"]}]}


def decide(body):
    if not gate.acquire(blocking=False):
        raise HTTPException(429, "Inference is busy; try later")
    try:
        return app.state.engine.decide(body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        gate.release()


@app.post("/v1/systemone")
async def systemone(request: Request):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 131072:
            raise HTTPException(413, "Request exceeds 128 KiB")
    try:
        body = json.loads(data)
        validate_request(body, CONFIG["max_questions"])
        if body.get("model", CONFIG["served_model"]) != CONFIG["served_model"]:
            raise ValueError("Requested model is not loaded")
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return await run_in_threadpool(decide, body)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, log_level="info")
