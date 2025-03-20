# JSONSchemaBench

[![ArXiv](https://img.shields.io/badge/arXiv-2501.10868-b31b1b)](https://arxiv.org/abs/2501.10868)
[![Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-orange)](https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench)


Reliably generating structured outputs is a key capability for modern LLM applications. Despite its growing adoption, a systematic evaluation of structured output generation is still lacking. With JSON Schema emerging as the standard format for structured data, we introduce JSONSchemaBench a benchmark of around 10,000 real-world JSON schemas that capture a wide range of constraints and complexities.
JSONSchemaBench helps to measure **efficiency** and **coverage** of a given structured output engine. 

<p align="center">
    <img src="img/hero_image.jpg" height="550"/>
       <br/>
    <em>Figure 1: Comparison across various constrained-decoding frameworks by efficiency (speed of output generation), coverage (support for JSON Schema features), and quality (effects on underlying task accuracy).</em>
</p>

## JsonSchemaBench

Visit [epfl-dlab/jsonschemabench](https://github.com/epfl-dlab/jsonschemabench) for the codebase of JSONSchemaBench.

## MaskBench

On top of JSONSchemaBench's view on real world JSON schema, we also develop a purely performance-oriented benchmark [MaskBench](./maskbench) targeting mask computation times. This benchmark emphasizes results relevant to server-side deployments of constrained decoding.

<p align="center">
    <img src="maskbench/plots/hero.png" width="700"/>
       <br/>
    <em>Figure 2: Isolated performance of token mask computation (for server-side scenarios). 
    See <a href="maskbench">MaskBench folder</a>.
    </em>
</p>

## Dataset Details

JSONSchemaBench is built from a collection of real-world JSON schemas drawn from diverse sources, including GitHub, Kubernetes configurations, and API specifications. The benchmark consists of schemas categorized into datasets based on complexity and domain. We start from collections from [json-schema-corpus](https://github.com/sdbs-uni-p/json-schema-corpus) and did heavy curation to ensure the schemas are standard-compliant and satisfiable. We also added schemas from other sources to increase the diversity of the benchmark, such as [GlaiveAI function call schemas](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) and [kubernetes schemas](https://github.com/instrumenta/kubernetes-json-schema).
We then categorized the schemas into datasets based on complexity and domain. The datasets are as follows:

<div align="center">

| Dataset         | Category            | Count |
| --------------- | ------------------- | ----- |
| GlaiveAI-2K     | Function Call       | 1707  |
| Github-Trivial  | Misc                | 444   |
| Github-Easy     | Misc                | 1943  |
| Snowplow        | Operational API     | 403   |
| Github-Medium   | Misc                | 1976  |
| Kubernetes      | Kubernetes API      | 1064  |
| Washington Post | Resource Access API | 125   |
| Github-Hard     | Misc                | 1240  |
| JSONSchemaStore | Misc                | 492   |
| Github-Ultra    | Misc                | 164   |
| **Total**       |                     | 9558  
</div>

For statistics on the datasets and an overview of schema constraint features, please refer to the [paper]()(link coming soon).

<details>
<summary>View Statistics on the Datasets</summary>

<p align="center">
    <img src="img/top_formats_pie_chart.png" height="300"/>
    <img src="img/top_features_bar_chart.png" height="300"/>
</p>

</details>

### 📥 Accessing the Dataset on Hugging Face
JSONSchemaBench is now available on the Hugging Face Hub. You can load it directly using the `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("epfl-dlab/JSONSchemaBench")
print(dataset)  
```
Each dataset split contains:
- `"json_schema"`: The schema definition.
- `"unique_id"`: A unique identifier for the schema.



## Data files

```
data
├── Github_easy
├── Github_hard
├── Github_medium
├── Github_trivial
├── Github_ultra
├── Glaiveai2K
├── JsonSchemaStore
├── Kubernetes
├── Snowplow
└── WashingtonPost
```





## JSON Schema Test Suite

For more details about the JSON Schema Test Suite used in the paper, visit the [official repository]((https://github.com/json-schema-org/JSON-Schema-Test-Suite)). 
The results of the test suite coverage are shown below

<details>
<summary>View Test Suite Coverage</summary>

<p align="center">
    <img src="img/Test_Suite_coverage.png" width="600"/>
</p>

</details>

## Feature Checklist

We provide a feature checklist for each Structured Output Generation Engine based on their documentation and implementation. This provides a comprehensive overview of the supported JSON Schema features.

<details>
<summary>View Feature Checklist</summary>

<p align="center">
    <img src="img/feature_checklist.png" width="600"/>
</p>

</details>

## Citation

```bibtex
@misc{geng2025jsonschemabench,
      title={Generating Structured Outputs from Language Models: Benchmark and Studies}, 
      author={Saibo Geng and Hudson Cooper and Michał Moskal and Samuel Jenkins and Julian Berman and Nathan Ranchin and Robert West and Eric Horvitz and Harsha Nori},
      year={2025},
      eprint={2501.10868},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.10868}, 
}
```
