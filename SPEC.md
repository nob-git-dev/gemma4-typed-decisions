# Gemma 4 Typed Decisions: specification

## Purpose

Provide typed decisions from the pinned official Gemma 4 checkpoint without generating text.
Compare accuracy and HTTP latency with the pinned official Jev service using human-labeled examples.
Reference prior ideas while independently implementing the decision layer and evaluation harness.

## Model and inference

- Model: `google/gemma-4-26B-A4B-it`.
- Revision: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.
- Final profile: `config.final.json`; generic vLLM FP8 loading, raw LAST pooling.
- Candidate readout: 26 original BF16 output rows, Gemma logit softcap, candidate softmax at temperature 1.
- Noul uses one ordering. Choice/Score use original and reversed option assignments in a batch,
  align by option name, and average logits before softmax.
- No fine tuning, generated text, third-party Jev reproduction implementation, derived weights,
  learned classification head, or Jev-generated training labels.
- Probabilities are uncalibrated and conditional on the provided alternatives.

## API contract

`POST /v1/systemone` accepts `state`, named `questions`, and an optional matching `model`.
State is a string, object, or array. Questions support:

- `noul`: true probability; optional definitions for true/false.
- `choice`: 2–26 named alternatives; chosen alternative and full candidate distribution.
- `score`: 2–10 ordered levels; expected index and full distribution.

`GET /health` and `GET /v1/models` describe the service.
This API does not provide free-form chat generation.
Malformed requests or model mismatches return 422, oversized HTTP bodies return 413,
and a concurrent request while inference is occupied returns 429.

Limits: 128 KiB HTTP request, 64 KiB serialized state, 16 logical questions,
4,096 input tokens per prompt, 16,384 tokens after option-order expansion.
The runtime rejects excess tokens without truncating them.

## Acceptance and evidence

The final paired comparison requires all of the following for the aggregate and each question type:

1. Candidate accuracy is at least 0.95 times Jev accuracy.
2. Candidate mean and p95 HTTP latency are each no greater than 1.05 times Jev latency.
3. The one-sided 95% paired bootstrap upper bound for relative accuracy loss is at most 0.05.
4. The corresponding upper bound for mean latency ratio is at most 1.05.
5. Each question type has at least 400 unique, human-labeled held-out inputs.

The bootstrap uses 500 repetitions and seed 20261027.
Accuracy is measured against public human labels, not agreement with Jev.
Score accuracy uses the modal class; normalized MAE uses the returned scalar score.
Jev displays rounded probabilities, so baseline validation explicitly handles its 0.01 rounding grid.
Reported scalar/distribution disagreements are audited without rewriting the vendor response.
Candidate distributions and expected scores remain strictly validated.

The comparison alternates endpoint order, reuses HTTP connections, excludes three warmups,
and performs no automatic retries. Wrong-model responses, missing labels, invalid distributions,
insufficient data, development data, and synthetic smoke examples cannot yield final PASS.

Development examples are separate from the original test and the fresh final test.
The final profile was selected using development data before examining fresh final predictions.
The final 3,300-pair run completed without reusing previous requests or resuming an interrupted run.

## Operating constraints

Docker targets ARM64 and binds the API to loopback only. Startup requires 116 GiB available RAM.
The final profile stops on available RAM below 36 GiB, GPU temperature at or above 80℃,
thermal zone at or above 90℃, or failed resource sensing.
It uses a 5 GiB loading buffer, a 50% PyTorch allocation cap, and a 4 GiB KV cache.
Other services are not automatically stopped; restart policy is `no`.

Credentials are used only by the official comparison client, which only sends them to the pinned
official HTTPS endpoint. Model inference does not require a TypeSafe key.

## Scope of the result

The recorded PASS applies to English SNLI entailment, AG News classification, and SST-5 sentiment
on the recorded GX10 environment with a warm model and compiler cache.
It does not establish Japanese accuracy, probability calibration, cold-start latency,
multi-client throughput, long-context scaling, multimodal support, or full Jev equivalence.
