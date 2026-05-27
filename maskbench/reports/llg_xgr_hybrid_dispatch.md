# LLGuidance/XGrammar hybrid dispatch

Date: 2026-05-27

## Scope

This experiment targets constrained-decoding infrastructure. It does not run an
LLM, train weights, or score model predictions. The benchmark is MaskBench token
mask computation over JSON-schema-constrained decoding traces.

The goal is to combine two existing strengths:

- XGrammar has very low steady-state mask latency on simple schemas.
- LLGuidance has the stronger correctness/coverage profile on complex schemas.

The implemented `--llg-xgr-hybrid` engine statically dispatches each schema:

- Use XGrammar compliant mode only when the schema is inside a conservative
  subset.
- Otherwise fall back to the LLGuidance reusable raw byte mask adapter.

## Safe Subset

A schema is dispatched to XGrammar only when all of these hold:

- No high-risk JSON Schema keywords are present, including combinators, `$ref`,
  regex/pattern constraints, format constraints, numeric bounds, string length
  bounds, array length bounds, and conditional/dependency keywords.
- Every `type` is a single primitive JSON type supported by the adapter.
- `additionalProperties`, when present, is `false`.
- For every object schema, `properties` is a dict, `required` is a list,
  `required` is a subset of `properties`, and every declared property is
  required.

The `required ⊆ properties` check is needed because XGrammar accepted a missing
required field when the required key had no declared property schema.

## Commands

```bash
python scripts/run_maskbench.py \
  --llg-rawbytes --output tmp/llgraw175-glaiveall \
  --time-limit 120 --num-threads 8 data/Glaiveai2K---*.json

python scripts/run_maskbench.py \
  --llg-xgr-hybrid --output tmp/hybrid175-glaiveall \
  --time-limit 120 --num-threads 8 data/Glaiveai2K---*.json

python scripts/run_maskbench.py \
  --llg-rawbytes --output tmp/llgraw175-full \
  --time-limit 900 --num-threads 8 data/*.json

python scripts/run_maskbench.py \
  --llg-xgr-hybrid --output tmp/hybrid175-full-v2 \
  --time-limit 900 --num-threads 8 data/*.json
```

Environment:

- `llguidance==1.7.5`
- `xgrammar==0.2.1`
- Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct`
- Local macOS workstation, 8 benchmark worker processes

## GlaiveAI Function-Call Split

The hybrid dispatches 763 of 1,707 schemas to XGrammar and preserves the
LLGuidance correctness profile.

| metric | `llg-rawbytes` | `llg-xgr-hybrid` | delta |
|:--|--:|--:|--:|
| passing schemas | 1,639 | 1,639 | same |
| compile errors | 68 | 68 | same |
| validation errors | 0 | 0 | same |
| invalidation errors | 0 | 0 | same |
| TBM avg | 50 us | 15 us | -70.0% |
| TBM p50 | 17 us | 6 us | -64.7% |
| TBM p75 | 34 us | 19 us | -44.1% |
| TBM p90 | 71 us | 36 us | -49.3% |
| TBM p95 | 108 us | 52 us | -51.9% |
| TBM p99 | 294 us | 135 us | -54.1% |
| TBM p99.9 | 3,651 us | 396 us | -89.2% |
| max mask | 122,950 us | 1,884 us | -98.5% |

Compared with XGrammar alone on the same split, the hybrid keeps most of the
steady-state speed but fixes the correctness profile:

| metric | `xgr-compliant` | `llg-xgr-hybrid` |
|:--|--:|--:|
| passing schemas | 1,561 | 1,639 |
| validation errors | 23 | 0 |
| invalidation errors | 123 | 0 |
| TBM avg | 17 us | 15 us |
| TBM p99 | 126 us | 135 us |

## Full Local Corpus

The hybrid dispatches 1,527 of 11,306 schemas to XGrammar and preserves the
LLGuidance correctness profile. The broad full-corpus speedup is real but small.

| metric | `llg-rawbytes` | `llg-xgr-hybrid` | delta |
|:--|--:|--:|--:|
| passing schemas | 9,487 | 9,487 | same |
| compile errors | 1,797 | 1,797 | same |
| validation errors | 22 | 22 | same |
| invalidation errors | 0 | 0 | same |
| TBM avg | 34 us | 34 us | flat |
| TBM p90 | 42 us | 40 us | -4.8% |
| TBM p95 | 72 us | 67 us | -6.9% |
| TBM p99 | 432 us | 417 us | -3.5% |
| TBM p99.9 | 1,629 us | 1,465 us | -10.1% |
| max mask | 159,051 us | 103,003 us | -35.2% |

## Claim Quality

This is a strong result for function-call schemas and a correctness-preserving
way to get XGrammar-like speed on a statically safe subset. It is not yet a big
full-corpus MaskBench state-of-the-art result.

The most defensible public framing is:

> A conservative LLGuidance/XGrammar hybrid preserves LLGuidance correctness on
> MaskBench's GlaiveAI function-call split while cutting average mask latency by
> 70% and p99.9 latency by 89%.

Relevant upstream context:

- MaskBench: https://github.com/guidance-ai/jsonschemabench/tree/main/maskbench
- LLGuidance: https://github.com/guidance-ai/llguidance
- XGrammar: https://github.com/mlc-ai/xgrammar
