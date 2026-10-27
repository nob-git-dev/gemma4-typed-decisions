# Recorded results — 2026-10-28

The final held-out comparison passed the predefined 5% gates on the recorded English tasks.
These are measured results from one environment, not a general claim of Jev equivalence.

## Final paired test

| Type | Pairs | Gemma correct / accuracy | Jev correct / accuracy | Gemma mean / p95 | Jev mean / p95 | Accuracy-loss upper 95% |
|---|---:|---:|---:|---:|---:|---:|
| Noul | 800 | 727 / 90.875% | 745 / 93.125% | 109.5 / 113.9 ms | 144.3 / 180.0 ms | 4.53% |
| Choice | 1,600 | 1,343 / 83.938% | 1,386 / 86.625% | 130.1 / 138.8 ms | 155.8 / 198.4 ms | 4.31% |
| Score | 900 | 505 / 56.111% | 507 / 56.333% | 123.7 / 129.0 ms | 142.6 / 168.3 ms | 4.61% |
| All | 3,300 | 2,575 / 78.030% | 2,638 / 79.939% | 123.4 / 135.1 ms | 149.4 / 185.8 ms | 3.50% |

Overall relative accuracy loss: 2.39%. Mean HTTP time reduction: 17.45%; p95 reduction: 27.29%.
All type-specific mean-latency ratio upper 95% bounds were at most 0.881.
See [summary.json](results/final/summary.json) for full-precision values and gates.

The reported aggregate Brier metric was 0.10486 for Gemma and 0.07283 for Jev (lower is better).
Its binary/multiclass definitions are in `metrics.py`; this aggregate mixes three task types.
Candidate probabilities are uncalibrated, and calibration equivalence is not established.
Score normalized MAE was 0.12394 versus 0.12293.
One official Jev scalar score/distribution discrepancy exceeded what rounding alone explained.
It was retained as a diagnostic; accuracy and MAE used the predefined scoring rules.

## Measurement conditions

- Candidate: official Gemma 4 26B A4B IT at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`,
  loaded as FP8 with original BF16 candidate readout; final two-order logit-mean policy.
- Baseline: official HTTPS `jev-1.14.0` service.
- Hardware: ASUS Ascent GX10, NVIDIA GB10, ARM64, 128 GB unified memory.
- Persistent HTTP connections, alternating endpoint order, three excluded warmups per backend,
  no retries, no resumed or reused requests in this final run.
- Client HTTP wall time includes the different local and cloud network paths.
  The model and compiler cache were warm; startup/download/compilation time is excluded.
- One logical question per request. Multi-question throughput and concurrent-load performance
  were not part of this final comparison.
- Correctness uses original human labels. Score accuracy is modal-class exact match.
- 500 paired bootstrap repetitions, seed 20261027; gates are recorded in [SPEC.md](SPEC.md).

## Data separation and development

Final test SHA256: `ea94491335c09d0accc898f7daee49d40be301b973acd4d01adec3353669610b`.
The final set contains 800 SNLI, 1,600 AG News, and 900 SST-5 test examples, with fixed label quotas.
Exact inputs from the original 1,200-example test and separate 1,200-example development set
were excluded. Dataset revisions are pinned in `prepare_eval.py`.

The initial single-order FP8 run had acceptable point estimates but failed to establish
all accuracy confidence bounds within 5% (Choice upper bound 7.19%, Score 5.31%).
Development-only selection then chose two-order logit averaging: Choice improved from
86.75% to 89.0%, and Score from 54.5% to 58.25% on development data.
The policy was fixed before evaluating the fresh final test.
[selection.json](results/final/selection.json) preserves this decision and its rule.

## Published evidence

- [summary.json](results/final/summary.json): unchanged final aggregate and type-specific statistics.
- [observations.jsonl](results/final/observations.jsonl): 6,600 numerical observations with public
  dataset row IDs, including paired correctness, latency, and probability error metrics.
- [manifest.json](results/final/manifest.json): model/config pins, original measured source hashes,
  dataset hash, measurement rules, and publication changes.
- [resources.json](results/final/resources.json): 490 resource samples summarized as minimum
  available RAM 77.95 GiB, maximum GPU 74℃, maximum thermal 82.7℃, and zero safety stops.
- [runtime.json](results/final/runtime.json): image identity and numerical runtime constraints.

The public export omits input text, raw API responses, host identity, and absolute local paths.
Numerical observations let readers recompute the published statistics, but cannot independently
re-score model answers against input text. Data preparation and live comparison scripts are included
for a new end-to-end evaluation. See [PRIVACY.md](PRIVACY.md).

```sh
uv run --no-project --python 3.12 tools/verify_results.py
```

The inference and statistical code is unchanged from the measured run. Publication changes only
adjust checkout-local cache paths, remove host identity and absolute dataset paths from future
benchmark manifests, and prepare standalone documentation.
