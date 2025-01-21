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
This particularly applies to XGrammar.

## Reproducing

Masks can be computed using `scripts/run_maskbench.py`.
Typical execution is `./scripts/run_maskbench.py --xgr-compliant data/`,
which will store the results under `tmp/out--xgr-compliant` folder.
See `./scripts/run_maskbench.py --help` for more details.
In particular, the resource limits can be adjusted via command line flags.

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
