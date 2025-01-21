# MaskBench

While the paper and the top-level repo here are mainly concerned with end-to-end performance and accuracy of
constrained generation with JSON Schema, this folder contains scripts and results
of benchmarking of just the mask computation, without any involvement of an LLM.

This lets us measure the performance of the mask computation in isolation, and also
is more applicable to server-side scenarios, where large batch sizes are used.

In particular, the `data/` folder contains one file per schema (around 10k),
and in most of them (8.5k) there are valid and invalid schema instances that can be used for
benchmarking and light-weight correctness testing.
In total there is 13k valid and 23k invalid instances, which works out to about 2M tokens.
See [below](#data-sources) for how these files were generated.

Thus, `data/` contains about 10k schemas, 13k valid and 23k invalid instances,
with 1.5k schemas having no tests,
for a total of about 2M tokens.

## Results

<p align="center">
    <img src="plots/tbm.png" />
</p>

<p align="center">
    <img src="plots/ttfm.png" />
</p>

## Engines

We have run the following grammar engines:

- [LLGuidance](https://github.com/guidance-ai/llguidance)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) grammars,
  with [json_schema_to_grammar.py](https://github.com/ggerganov/llama.cpp/blob/master/examples/json_schema_to_grammar.py)
  with whitespace regex modified to `/[ \t\n\r]*/` to match JSON definition
  (by default it puts limits on the amount of whitespace, which slows down the engine)
- [XGrammar](https://github.com/mlc-ai/xgrammar) in default configuration
- "XGrammar.cpp", which is XGrammar engine together with the above script
- [Outlines](https://github.com/dottxt-ai/outlines)

## Test setup

All tests were run on
[Azure NC96ads_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic)
with AMD EPYC 7V13 (Milan) with 96 threads enabled (48 cores),
with 880 GiB of RAM and 4xA100 80GB PCIe GPUs (though GPUs were not used).

There was a time limit of 15 minutes per schema,
memory limit of 40GiB resident set size (which works out to a few GiB of working set)
and a 40 running thread limit (these are all defaults, except for thread limit which
will be lower on lesser machines).

We run every engine single-threaded, under the assumption that for heavy
production work-loads, the batch size will be bigger than the number of
available cores, so it makes sense to run each sequence in a single thread.
We've set XGrammar to only use one thread and LLGuidance always uses one thread.
Outlines seems to be using more than one thread, so we instead run it with 90
threads in parallel to limit the CPU this way.

## Discussion

Grammar compilation time (TTFM) is almost instantaneous for LLGuidance and llama.cpp
(the high p100 time for llama.cpp is likely due to Python implementation of JSON Schema conversion).
Outlines has by far the highest TTFM, and this doesn't account for over 1000 timeouts (15 minutes or more).
Observed XGrammar's TTFM can likely often be hidden in prefill time, provided the computation is parallelized,
but it's consistently 3-4 orders of magnitude slower than LLGuidance and llama.cpp.

For mask computation time (TBM), XGrammar in its default settings beats LLGuidance
up to p75 (the easy cases), but becomes 2x slower at p90 and 10x slower at p99.
Due to this slow tail, LLGuidance is 6x faster than XGrammar on average.

When a comparable grammar is considered (XGrammar.cpp), XGrammar is much slower
than LLGuidance (2-3 orders of magnitude) from p50 onwards,
and 3 orders of magnitude slower on average.
Interestingly, llama.cpp using the same grammar as XGrammar is in fact faster
on average and by 1 order of magnitude above p75.

Outlines, despite having everything pre-computed, is very slow.

## Random notes

- for TBM, with batch size 100 and forward pass time of 20ms, the p99 happens 50 times per second,
  and p99.9 happens 5 times per second; unless handled specially, these mask computations
  will hold the entire batch
- the TTFM is cut off at 900s due to timeout (especially for Outlines)
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



<!-- GEN-BEGIN -->
| metric             | LLGuidance | XGrammar (defl.) |  llama.cpp | XGrammar.cpp |    Outlines |
|:-------------------|-----------:|-----------------:|-----------:|-------------:|------------:|
| TBM avg (us)       |         47 |              292 |     14,523 |       64,805 |      59,420 |
| TBM p25            |         16 |                3 |         87 |            7 |          16 |
| TBM p50            |         36 |                8 |     14,483 |          866 |         118 |
| TBM p75            |         43 |               54 |     20,038 |      123,211 |     133,981 |
| TBM p90            |         60 |              117 |     22,743 |      185,712 |     200,689 |
| TBM p95            |         98 |              294 |     25,111 |      242,835 |     243,995 |
| TBM p99            |        460 |            4,936 |     57,317 |      615,009 |     358,041 |
| TBM p99.9          |      1,453 |           44,687 |    268,292 |    1,923,424 |     598,956 |
| TBM p100           |     20,007 |          694,664 |  1,083,198 |    7,975,547 |   1,260,641 |
| TTFM avg (us)      |      1,611 |        4,137,527 |      3,117 |    3,347,141 |  37,948,447 |
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

## Reproducing

Masks can be computed using `scripts/run_maskbench.py`.
Typical execution is `./scripts/run_maskbench.py --xgr-compliant data/`,
which will store the results under `tmp/out--xgr-compliant` folder.
See `./scripts/run_maskbench.py --help` for more details.
In particular, the resource limits can be adjusted via command line flags.

Once the results are generated, you can generate table and plots with
`./scripts/maskbench_results.py`.

## Data sources

The schema instances instances were generated with Meta Llama 3.1 70B instruct model,
constrained to output valid JSON, but not the schema.
If a valid instance was generated, the model was additionally prompted to
modify it to make it invalid.
The prompts were tuned to have the model focus on specific features
of the schema (e.g. `maxItems` or `pattern`),
and some were generated without such focus.
The data generation scripts are under `creation/` folder.

Both valid and invalid instances were checked by Python and Rust jsonschema libraries.
