# MaskBench

<p align="center">
    <img src="plots/hero.png" width="700"/>
</p>

The primary focus of this repository is end-to-end performance and accuracy of JSON Schema-constrained generation. This folder, however, contains scripts and results dedicated to benchmarking mask computation in isolation, without involving an LLM.

By isolating mask computation, this benchmark assesses its standalone performance, which is particularly relevant for server-side scenarios with large batch sizes.

### Data Overview

- **Data Folder (`data/`)**: Contains ~10k schemas, with 13k valid and 23k invalid instances (total: ~2M tokens). About 1.5k schemas lack tests.  
- **Schema Instances**: Each schema includes valid and invalid examples for benchmarking and correctness testing. See [Data Sources](#data-sources) for details on data generation.

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
5. **[Outlines](https://github.com/dottxt-ai/outlines)**

## Test Environment

- **Hardware**: Azure [NC96ads_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic) with 96 threads (48 cores), 880 GiB RAM, 4x A100 GPUs (GPUs not utilized).
- **Constraints**: 
  - Time: 15 minutes per schema.
  - Memory: 40 GiB resident set size.
  - Threads: 40-thread limit.

- Engines were executed single-threaded to emulate large batch scenarios (where batch size is larger than the number of available cores).
- XGrammar was set to only use a single thread per sequence, LLGuidance and llama.cpp always do that.
- Outlines normally uses several threads per sequence, so it was run with 90 parallel threads, so it doesn't get more CPU time than the other engines.

## Key Findings

- **Grammar Compilation Time (TTFM)**:
  - **LLGuidance** and **llama.cpp** had near-instantaneous compilation.
  - **Outlines** was the slowest, with 1000+ timeouts (15+ minutes).
  - **XGrammar**'s TTFM is 3–4 orders of magnitude slower than **LLGuidance** and **llama.cpp**. It's likely grammar compilation time can be hidden in prefill time (especially when parallelized).

- **Mask Computation Time (TBM)**:
  - **XGrammar** outperforms **LLGuidance** on simple cases (p25–p75), but becomes 2x and 10x slower at p90 and p99, respectively.
  - **LLGuidance** is thus 6x faster on average due to better tail performance.
  - **Outlines** is very slow despite pre-computed results.
  - **XGrammar.cpp** lags significantly (2–3 orders of magnitude slower than **LLGuidance** from p50 onward).

## Random notes

- for TBM, with batch size 100 and forward pass time of 20ms, the p99 happens 50 times per second,
  and p99.9 happens 5 times per second; unless handled specially, these mask computations
  will hold the entire batch
- the TTFM is cut off at 900s due to timeout (especially relevant for Outlines)
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
| metric             | LLGuidance | XGrammar (defl.) |  llama.cpp | XGrammar.cpp |    Outlines |
|:-------------------|-----------:|-----------------:|-----------:|-------------:|------------:|
| TBM avg            |         47 |              292 |     14,523 |       64,805 |      59,420 |
| TBM p25            |         16 |                3 |         87 |            7 |          16 |
| TBM p50            |         36 |                8 |     14,483 |          866 |         118 |
| TBM p75            |         43 |               54 |     20,038 |      123,211 |     133,981 |
| TBM p90            |         60 |              117 |     22,743 |      185,712 |     200,689 |
| TBM p95            |         98 |              294 |     25,111 |      242,835 |     243,995 |
| TBM p99            |        460 |            4,936 |     57,317 |      615,009 |     358,041 |
| TBM p99.9          |      1,453 |           44,687 |    268,292 |    1,923,424 |     598,956 |
| TBM p100           |     20,007 |          694,664 |  1,083,198 |    7,975,547 |   1,260,641 |
| TTFM avg           |      1,611 |        4,137,527 |      3,117 |    3,347,141 |  37,948,447 |
| TTFM p25           |        792 |          432,583 |        209 |      240,697 |   3,928,969 |
| TTFM p50           |        987 |          535,498 |        307 |      534,235 |   6,987,012 |
| TTFM p75           |      1,402 |        1,043,954 |        671 |      971,319 |  16,556,110 |
| TTFM p90           |      2,584 |        2,996,171 |      1,909 |    2,827,280 |  78,684,322 |
| TTFM p95           |      4,408 |        7,379,516 |      4,475 |    6,918,368 | 206,301,765 |
| TTFM p99           |     14,180 |       62,151,076 |     31,263 |   41,703,619 | 621,693,634 |
| TTFM p99.9         |     24,604 |      521,102,130 |    179,470 |  424,413,135 | 853,211,644 |
| TTFM p100          |    170,635 |      837,805,312 | 10,076,300 |  517,111,658 | 888,396,817 |
| tokens             |  2,565,248 |        2,096,637 |  2,035,114 |    1,461,592 |   1,042,656 |
| schemas            |     10,163 |           10,163 |     10,163 |       10,163 |      10,163 |
| passing            |      7,765 |            5,216 |      5,476 |        5,356 |       4,259 |
| compile error      |      2,373 |            2,089 |      1,292 |        1,709 |       3,608 |
| segmentation fault |          0 |              189 |          0 |            1 |           0 |
| out of memory      |          0 |                0 |          0 |            0 |          13 |
| timeout            |          0 |               11 |          1 |           48 |       1,020 |
| validation error   |         25 |            1,409 |      2,745 |        2,726 |         649 |
| invalidation error |          0 |            1,249 |        649 |          323 |         614 |


### Versions

* llguidance: 0.6.7
* xgrammar: 0.1.10
* llama-cpp-python: 0.3.6
* outlines: 0.1.13
<!-- GEN-END -->

## Reproducing Results

- **Run Masks**: Use `scripts/run_maskbench.py`. Example:  
  `./scripts/run_maskbench.py --xgr-compliant data/`  
  Results are saved in `tmp/out--xgr-compliant`.
  See `./scripts/run_maskbench.py --help` for more options, in particular resource limits.
  
- **Analyze Results**: Generate tables and plots with  
  `./scripts/maskbench_results.py`.

## Data Sources

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