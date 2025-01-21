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

We have run the following grammar engines:

- [llguidance](https://github.com/guidance-ai/llguidance) v0.6.7
- [XGrammar](https://github.com/mlc-ai/xgrammar) v0.1.10 in default configuration
- XGrammar with flexible white-space and "non-strict" mode (ie., compliant, or adhering to JSON Schema specification)
- [Outlines](https://github.com/dottxt-ai/outlines) v0.1.13

All tests were run on
[Azure NC96ads_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic)
with AMD EPYC 7V13 (Milan) with 96 threads enabled (48 cores),
with 880 GiB of RAM and 4xA100 80GB PCIe GPUs (though GPUs were not used).

There was a time limit of 15 minutes per schema,
memory limit of 40GiB resident set size (which works out to a few GiB of working set)
and a 90 running thread limit (these are all defaults, except for thread limit which
will be lower on lesser machines).

We run every engine single-threaded, under the assumption that for heavy
production work-loads, the batch size will be bigger than the number of
available cores, so it makes sense to run each sequence in a single thread.
We've set XGrammar to only use one thread and LLGuidance always uses one thread.
Outlines seems to be using more than one thread, so we instead run it with 90
threads in parallel to limit the CPU this way.

Random notes:

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
  however, all engines stick to definition order in `properties`,
  and engines other than LLGuidance don't support `allOf` and sibling properties
  (which introduces complications to the ordering)
- XGrammar in "compliant" (non-default) configuration, doesn't apply constraints
  to non-required properties (for object with properties `"a"` and `"b"` it uses grammar
  similar to `("a": A)? ("b": B)? ("[^"]+": ANYTHING)*`, so in object `{"a":1}`
  `1` can be matched to **either** `A` or `ANYTHING`);
  this is why it has lots of invalidation errors and processes more tokens (since it gets further
  into the invalid instances)

<!-- GEN-BEGIN -->
| metric             | LLGuidance | XGrammar (defl.) |    XGrammar |    Outlines |
|:-------------------|-----------:|-----------------:|------------:|------------:|
| TBM avg (us)       |         49 |              292 |       5,671 |      59,420 |
| TBM p25            |         16 |                3 |          12 |          16 |
| TBM p50            |         36 |                8 |          80 |         118 |
| TBM p75            |         43 |               54 |         321 |     133,981 |
| TBM p90            |         61 |              117 |       1,668 |     200,689 |
| TBM p95            |        100 |              294 |       5,919 |     243,995 |
| TBM p99            |        468 |            4,936 |      66,378 |     358,041 |
| TBM p99.9          |      1,468 |           44,687 |     889,990 |     598,956 |
| TBM p100           |     66,397 |          694,664 |  30,748,937 |   1,260,641 |
| TTFM avg (us)      |      1,637 |        4,137,530 |   5,504,335 |  37,948,447 |
| TTFM p25           |        795 |          432,583 |     770,277 |   3,928,969 |
| TTFM p50           |        984 |          535,498 |     995,320 |   6,987,012 |
| TTFM p75           |      1,420 |        1,043,954 |   1,895,386 |  16,556,110 |
| TTFM p90           |      2,646 |        2,996,171 |   4,925,680 |  78,684,322 |
| TTFM p95           |      4,704 |        7,379,516 |  11,376,422 | 206,301,765 |
| TTFM p99           |     14,526 |       62,151,076 | 112,906,309 | 621,693,634 |
| TTFM p99.9         |     24,979 |      521,102,130 | 566,943,598 | 853,211,644 |
| TTFM p100          |    160,896 |      837,805,312 | 854,445,907 | 888,396,817 |
| tokens             |  2,565,234 |        2,096,637 |   2,709,854 |   1,042,656 |
| schemas            |     10,163 |           10,163 |      10,163 |      10,163 |
| passing            |      7,765 |            5,216 |       4,719 |       4,259 |
| compile error      |      2,373 |            2,089 |       2,091 |       3,608 |
| segmentation fault |          0 |              189 |         187 |           0 |
| out of memory      |          0 |                0 |           0 |          13 |
| timeout            |          0 |               11 |          83 |       1,020 |
| validation error   |         24 |            1,409 |         495 |         649 |
| invalidation error |          0 |            1,249 |       2,588 |         614 |
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
