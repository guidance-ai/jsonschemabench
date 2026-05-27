# LLGuidance fast-forward effective overhead on GlaiveAI

Date: 2026-05-27

## Scope

This experiment targets constrained-decoding infrastructure only. It does not run
an LLM, train weights, score model outputs, or use prediction accuracy. The
workload is the MaskBench GlaiveAI-2K split: 1,707 function-call JSON schemas.

MaskBench's standard TBM measures mask computation per generated token. This
variant measures effective serving overhead per generated token when deterministic
fast-forward tokens are emitted without computing a full mask.

## Implementation

`--llg-ff` adds an LLGuidance adapter that:

1. Calls `LLMatcher.compute_ff_tokens()` at each state.
2. If LLGuidance reports a deterministic token run, consumes that run with
   `consume_tokens()` and records the elapsed overhead distributed across those
   emitted tokens.
3. Otherwise falls back to the reusable raw byte mask path from
   `--llg-rawbytes`.

The correctness contract is unchanged: the same valid instances must be accepted
and the same invalid instances must be rejected.

## Commands

```bash
python scripts/run_maskbench.py \
  --llg-rawbytes --output tmp/llgraw175-glaiveall \
  --time-limit 120 --num-threads 8 data/Glaiveai2K---*.json

python scripts/run_maskbench.py \
  --llg-ff --output tmp/llgff175-glaiveall \
  --time-limit 120 --num-threads 8 data/Glaiveai2K---*.json
```

Environment:

- `llguidance==1.7.5`
- `xgrammar==0.2.1`
- Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct`
- Local macOS workstation, 8 benchmark worker processes

## Result

| metric | `llg-rawbytes` | `llg-ff` | delta |
|:--|--:|--:|--:|
| schemas passing | 1,639 | 1,639 | same |
| compile errors | 68 | 68 | same |
| validation errors | 0 | 0 | same |
| invalidation errors | 0 | 0 | same |
| tokens measured | 81,035 | 81,034 | same |
| full mask calls | 81,035 | 67,275 | -17.0% |
| fast-forward tokens | 0 | 13,278 | +16.39% |
| effective avg | 50 us | 27 us | -46.0% |
| effective p50 | 17 us | 15 us | -11.8% |
| effective p95 | 108 us | 88 us | -18.5% |
| effective p99 | 294 us | 189 us | -35.7% |
| effective p99.9 | 3,651 us | 1,084 us | -70.3% |
| summed effective overhead | 4,092,470 us | 2,232,804 us | -45.4% |

## Non-win On Full Corpus

On the full 11,306-schema local corpus, the same fast-forward adapter did not
produce a broad MaskBench win:

| metric | `llg-rawbytes` | `llg-ff` |
|:--|--:|--:|
| schemas passing | 9,487 | 9,487 |
| compile errors | 1,797 | 1,797 |
| validation errors | 22 | 22 |
| invalidation errors | 0 | 0 |
| effective avg | 34 us | 34 us |
| effective p99 | 432 us | 422 us |
| fast-forward token share | 0% | 10.84% |

So the public claim should not be "full MaskBench SOTA." The defensible claim is
that function-call schemas, represented by the GlaiveAI split, get a large
effective-overhead reduction while preserving LLGuidance's correctness profile.

## Rejected Variants

- Bounded prefix mask cache: improved GlaiveAI average less than fast-forward and
  made the combined prefix+fast-forward variant slower.
- Compact JSON replay: reduced token count, but worsened per-token overhead and
  did not beat default JSON plus fast-forward on total overhead.

## Publication Fit

This is worth sharing as a technical note or upstream issue/PR discussion if
framed as a serving-runtime metric for function-call schemas. It is not yet a
general SOTA result across all constrained-decoding workloads.

Relevant upstream context:

- MaskBench: https://github.com/guidance-ai/jsonschemabench/tree/main/maskbench
- LLGuidance: https://github.com/guidance-ai/llguidance
- XGrammar: https://github.com/mlc-ai/xgrammar
