# MaskBench

<p align="center">
    <img src="plots/hero.png" width="700"/>
</p>

The primary focus of the top-level repository is end-to-end performance and accuracy of JSON Schema-constrained generation.
See [paper](https://arxiv.org/abs/2501.10868) for general methodology, description of data and results.

This folder, however, contains scripts and results dedicated to benchmarking mask computation in isolation, without involving an LLM.
By isolating mask computation, this benchmark assesses its standalone performance, which is particularly relevant for server-side scenarios with large batch sizes.

## News

- **2024-03-27**: re-run tests with latest versions of engines; fairer timeout accounting
- **2024-03-26**: added testcases from [NousResearch/json-mode-eval](https://huggingface.co/datasets/NousResearch/json-mode-eval)
- **2024-03-26**: added (quite easy) testcases from [Gorilla BFCL v3](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/data) using [improved](./creation/fetch_bfcl.py) version of script by @zanderjiang
- **2024-03-21**: reordered properties in a few objects, to follow [stable property order](https://github.com/guidance-ai/llguidance/blob/main/docs/json_schema.md#property-order)
- **2024-02-17**: re-run tests with latest versions of engines
- **2024-01-20**: initial release of the benchmark

## Data Overview

- **Data Folder (`data/`)**: Contains ~10k schemas, with 13k valid and 23k invalid instances (total: ~2M tokens). About 1.5k schemas lack tests.  
- **Schema Instances**: Each schema includes valid and invalid examples for benchmarking and correctness testing. See [Testcases](#testcases) for details on data generation and stats on the testcases.

## Benchmark Results

<p align="center">
    <img src="plots/tbm.png" />
</p>

<p align="center">
    <img src="plots/ttfm.png" />
</p>

## Engines Benchmarked

1. **[LLGuidance](https://github.com/guidance-ai/llguidance)**
2. **[llama.cpp](https://github.com/ggerganov/llama.cpp)** grammars,
  with [json_schema_to_grammar.py](https://github.com/ggerganov/llama.cpp/blob/master/examples/json_schema_to_grammar.py)
  with whitespace regex modified to `/[ \t\n\r]*/` to match JSON definition
  (by default it puts limits on the amount of whitespace, which slows down the engine).
3. **[XGrammar](https://github.com/mlc-ai/xgrammar)** in default configuration.
4. **"XGrammar.cpp"**: XGrammar with the llama.cpp script above.
5. **[Outlines Core](https://github.com/dottxt-ai/outlines-core)**

## Test Environment

- **Hardware**: Azure [NC96ads_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic) with 96 threads (48 cores), 880 GiB RAM, 4x A100 GPUs (GPUs not utilized).
- **Constraints**: 
  - Time: 15 minutes per schema.
  - Memory: 40 GiB resident set size.
  - Threads: 40-thread limit.

- Engines were executed single-threaded to emulate large batch scenarios (where batch size is larger than the number of available cores).
- XGrammar was set to only use a single thread per sequence, other always do that.
- ~~Outlines normally uses several threads per sequence, so it was run with 90 parallel threads, so it doesn't get more CPU time than the other engines.~~

Approximate times to run the benchmark with 40-way parallelism:
- LLGuidance: ~45 seconds (Python overhead; Rust benchmark takes ~7 seconds)
- llama.cpp: ~20 minutes
- XGrammar: ~80 minutes
- XGrammar.cpp: ~70 minutes
- Outlines: ~130 minutes

## Measurements

TTFM p75 is computed as follows:
take all grammar compilation times that did not raise exception for a given engine
(including timeouts, which count for the timeout value (900s)),
sort them, and take the 75th percentile.

TBM p75 is computed as follows: take all successful mask computation times for a given engine (accross all schemas), sort them, and take the 75th percentile.

Note that an engine that only supports "easy" schemas may have artificially good scores.

## Key Findings

- **Grammar Compilation Time (TTFM)**:
  - **LLGuidance** and **llama.cpp** had near-instantaneous compilation (skewed by the single llama.cpp timeout).
  - **XGrammar** is now the slowest (unlike in previous runs of this benchmark), likely because it now supports more JSON Schema features.
  - **Outlines** has improved dramatically, and is now slightly faster than **XGrammar** (though the feature comparison is unclear).
  - **XGrammar.cpp** is now somewhat faster than regular **XGrammar** for the more complex schemas.

- **Mask Computation Time (TBM)**:
  - **XGrammar** outperforms **LLGuidance** on simple cases (p25–p90), but becomes 4x and 11x slower at p95 and p99, respectively.
  - **LLGuidance** is thus 11x faster on average due to better tail performance. The average dropped in 0.7.10 because computing the mask now drops the GIL, which takes a few microseconds.
  - **Outlines** has fully pre-computed results, so it's unclear why it's so slow at p75 and above.
  - **XGrammar.cpp** lags significantly (2–3 orders of magnitude slower than **LLGuidance** from p50 onward).

## Random notes

- for TBM, with batch size 100 and forward pass time of 20ms, the p99 happens 50 times per second,
  and p99.9 happens 5 times per second; unless handled specially, these mask computations
  will hold the entire batch
- the TTFM is cut off at 900s due to timeout
- while LLGuidance has the biggest number of compile errors,
  it has almost no validation errors nor crashes;
  in other words it's upfront about what it cannot do
- the "invalidation errors" are cases where a generation should be rejected,
  but wasn't; these are clear bugs
- "validation errors" may be more tricky due to object property ordering;
  however, all engines stick to definition order in `properties`
  (except for llama.cpp, which puts required properties first),
  and engines other than LLGuidance don't support `allOf` and sibling properties
  (which introduces complications to the ordering)
- by default, XGrammar uses fixed white-space and assumes `"additionalProperties": false`;
  it also ignores keywords like `minItems` and `maxItems`, as well as `allOf`, sibling properties, etc.;
  this all significantly reduces complexity of the grammar
- OTOH, grammars used in XGrammar.cpp configuration are similar in feature coverage
  and flexibility to the ones used in LLGuidance; thus it provides a more
  apples-to-apples comparison of the grammar engines (as opposed to grammars)

## Performance Metrics

<!-- GEN-BEGIN -->
| metric             | LLGuidance |    XGrammar |   llama.cpp | XGrammar.cpp |    Outlines |
|:-------------------|-----------:|------------:|------------:|-------------:|------------:|
| TBM avg            |         64 |         728 |      16,565 |       42,158 |       1,858 |
| TBM p25            |         29 |           3 |      11,032 |            8 |           2 |
| TBM p50            |         46 |           9 |      17,021 |        1,677 |          34 |
| TBM p75            |         55 |          39 |      22,656 |       74,994 |       4,110 |
| TBM p90            |         77 |          99 |      26,818 |      106,677 |       6,043 |
| TBM p95            |        119 |         499 |      29,722 |      137,862 |       6,572 |
| TBM p99            |        533 |       5,687 |      60,371 |      381,153 |       7,184 |
| TBM p99.9          |      1,695 |     100,267 |     297,552 |    2,043,383 |       7,820 |
| TBM p100           |     33,055 |   6,283,834 |   1,129,560 |    7,416,822 |   1,121,673 |
| TTFM avg           |      1,850 |  27,062,414 |      92,371 |    7,040,547 |  18,351,800 |
| TTFM p25           |        800 |     683,313 |         220 |      434,100 |     300,638 |
| TTFM p50           |      1,037 |     919,459 |         312 |      856,231 |     587,900 |
| TTFM p75           |      1,534 |   2,399,718 |         621 |    1,445,446 |   4,281,360 |
| TTFM p90           |      3,030 |  11,645,219 |       1,744 |    4,099,282 |  26,703,408 |
| TTFM p95           |      5,457 |  35,854,415 |       3,649 |    9,323,880 |  58,638,471 |
| TTFM p99           |     17,988 | 900,000,000 |      28,558 |   65,476,826 | 413,111,222 |
| TTFM p99.9         |     40,092 | 900,000,000 |     112,446 |  900,000,000 | 900,000,000 |
| TTFM p100          |    229,367 | 900,000,000 | 900,000,000 |  900,000,000 | 900,000,000 |
| tokens             |  2,607,859 |   2,929,227 |   2,069,143 |    1,511,558 |   2,349,083 |
| schemas            |     11,306 |      11,306 |      11,306 |       11,306 |      11,306 |
| passing            |      8,909 |       8,122 |       6,505 |        6,387 |       7,050 |
| compile error      |      2,377 |          90 |       1,301 |        1,718 |       1,773 |
| segmentation fault |          0 |          37 |           0 |            1 |           0 |
| out of memory      |          0 |           0 |           0 |            0 |         188 |
| timeout            |          0 |         241 |           1 |           38 |          60 |
| validation error   |         20 |       1,408 |       2,850 |        2,837 |       1,415 |
| invalidation error |          0 |       1,368 |         649 |          325 |         822 |


### Versions

* llguidance: 0.7.10
* xgrammar: 0.1.17
* llama-cpp-python: 0.3.8
* outlines-core: 0.2.3
<!-- GEN-END -->

## Reproducing Results

- **Run Masks**: Use `scripts/run_maskbench.py`. Example:  
  `./scripts/run_maskbench.py --xgr-compliant data/`  
  Results are saved in `tmp/out--xgr-compliant`.
  See `./scripts/run_maskbench.py --help` for more options, in particular resource limits.
  
- **Analyze Results**: Generate tables and plots with  
  `./scripts/maskbench_results.py`.


### Debugging engine

Run `python -m maskbench.runner data/Github_easy---o13947.json --debug --llg`. Replace `--llg` with other engine options as needed.

## Testcases

The schema instances were generated using the Meta Llama 3.1 70B instruct model. The output was constrained to produce valid JSON, though not strictly conforming to the schema. For valid instances, the model was further prompted to modify them into invalid ones.

Prompts for invalid instances were adjusted to emphasize specific schema features (e.g., `maxItems`, `pattern`, `minLength`, `if` etc.), while some instances were generated without such focus. The data generation scripts are located in the `creation/` folder.

Both valid and invalid instances were validated using Python and Rust jsonschema libraries.

Tests are categorized by origin and complexity. The table below summarizes the number of schemas, the percentage with generated instances, and the count of valid and invalid instances (some schemas have multiple valid/invalid instances).

For valid instances only (since invalid instances are not generated in production), the following metrics are computed:

- **Average tokens per instance** (as counted by the Llama3 tokenizer).
- **Fast-forward token share**, measured for two cases:
  1. Regular JSON (allowing whitespace wherever permitted by the spec).
  2. Compacted JSON (no whitespace anywhere).

Compacted JSON is preferred unless the model is fine-tuned on indented JSON.

Fast-forward tokens are additional tokens that can be appended to the model's context window after sampling. These tokens are 3-10x faster to compute than regular tokens. For example, a 15% share of fast-forward tokens corresponds to a 10-13% increase in throughput.

| split           | schemas | has tests | valid inst. | invalid inst. | tok/inst. |  FF | FF compact |
|:----------------|--------:|----------:|------------:|--------------:|----------:|----:|-----------:|
| Github_trivial  |     444 |       73% |         460 |           771 |        41 |  3% |         5% |
| Github_easy     |    1943 |       87% |        2641 |          4611 |        46 | 11% |        14% |
| Github_hard     |    1240 |       68% |        1493 |          3405 |       339 | 16% |        19% |
| Github_medium   |    1976 |       87% |        3091 |          6119 |       141 | 11% |        13% |
| Github_ultra    |     164 |       54% |         160 |           302 |       768 | 19% |        21% |
| Glaiveai2K      |    1707 |       61% |        1634 |          1104 |        30 | 21% |        25% |
| Kubernetes      |    1064 |       89% |        1680 |          2908 |        86 |  9% |        10% |
| Snowplow        |     403 |       95% |         670 |          1730 |       142 |  9% |        11% |
| WashingtonPost  |     125 |       78% |         146 |           330 |        95 | 12% |        14% |
| MCPspec         |      45 |       78% |          44 |            44 |        45 | 20% |        29% |
| JsonSchemaStore |     492 |       73% |         679 |          1405 |       295 |  7% |         7% |
| TOTAL           |   10163 |       75% |       12821 |         23047 |       133 | 13% |        15% |