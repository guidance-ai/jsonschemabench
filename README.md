# JSONSchemaBench

Reliably generating structured outputs is a key capability for modern LLM applications. Despite its growing adoption, a systematic evaluation of structured output generation is still lacking. With JSON Schema emerging as the standard format for structured data, we introduce JSONSchemaBench a benchmark of around 10,000 real-world JSON schemas that capture a wide range of constraints and complexities.
JSONSchemaBench helps to measure **efficiency** and **coverage** of a given structured output engine. 

<p align="center">
    <img src="img/hero_image.jpg" height="550"/>
       <br/>
    <em>Figure 1: Comparison across various constrained-decoding frameworks by efficiency (speed of output generation), coverage (support for JSON Schema features), and quality (effects on underlying task accuracy).</em>
</p>

<p align="center">
    <img src="maskbench/plots/hero.png" width="550"/>
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

Each folder contains the json schema included 
```
data
└── Github_easy
    ├── track_calories_82bd0ec9.json
    ├── track_calories_93d75421.json
    ├── track_calories_be81cd27.json
    ├── track_calories_d61851c9.json
    ├── track_calories_d9be8839.json
    ├── track_calories_ecbd9766.json
    ├── track_expenses_a6fa070d.json
    ├── track_expenses_c8268204.json
    └── track_fitness_activity_2989efaf.json
```

## Quick Start

Step 1. Load a JSON Schema from the GlaiveAI Dataset:

```python
import json
with open('data/Glaiveai2K/search_restaurants_d4619845.json') as f:
    schema = json.load(f)
    print(schema)
```


<details>
<summary> View Example JSON Schema </summary>

```json
{
  "properties": {
    "cuisine": {
      "description": "The cuisine to search for",
      "type": "string"
    },
    "location": {
      "description": "The location to search for restaurants",
      "type": "string"
    },
    "price_range": {
      "properties": {
        "max_price": {
          "description": "The maximum price range for restaurants",
          "type": "number"
        },
        "min_price": {
          "description": "The minimum price range for restaurants",
          "type": "number"
        }
      },
      "required": [
        "min_price",
        "max_price"
      ],
      "type": "object"
    }
  },
  "required": [
    "location",
    "cuisine",
    "price_range"
  ],
  "type": "object"
}
 ```

</details>


Step 2. Create a prompt to generate structured output:

```python
prompt = f"Generate a function call that adheres to the following schema: {schema}"
# Feel free to include few-shot examples or additional context to improve the output.
```

Step 3. Use a Structured Output Generation Engine to generate output:

Below are examples for various structured output generation engines. For further details, refer to the official documentation of each engine.

### OpenAI

<details>
<summary>View Code Example</summary>

Requires `openai` library and a valid OpenAI API key. Install the library using:
```bash
pip install openai
```

```python
import os
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# OpenAI requires "additional_properties": false in the schema and all nested sub-schemas.
schema["additionalProperties"] = False
schema["properties"]["price_range"]["additionalProperties"] = False

openai_response = client.chat.completions.create(
  model="gpt-4o-2024-08-06",
  messages=[
    {"role": "user", "content": prompt}
  ],
  response_format={
    "type": "json_schema",
    "json_schema": {
      "name": "default",
      "strict": True,
      "schema": schema
    }
  }
)

output:str = openai_response.choices[0].message.content
```

</details>

### Gemini

<details>
<summary>View Code Example</summary>

Requires `google-generativeai` library and a valid Google API key. Install the library using:
```bash
pip install google-generativeai
```

```python
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content(contents=prompt, generation_config=genai.GenerationConfig(
        response_mime_type="application/json", response_schema=schema
    ))
output:str = response.text
```

</details>

### Guidance

<details>
<summary>View Code Example</summary>

Requires `guidance` and `transformers` libraries. Install them using:
```bash
pip install guidance transformers
```

```python
import guidance
operator = guidance.json(schema=schema, name='json_generation')
guidance_model = guidance.models.Transformers('meta-llama/Llama-3.2-1B-Instruct')

response = guidance_model + prompt + operator

output:str = response["json_generation"]
```

</details>

### XGrammar


<details>
<summary>View Code Example</summary>

Requires `xgrammar` and `transformers` libraries. Install them using:
```bash
pip install xgrammar transformers
```

```python
import xgrammar
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
    hf_tokenizer, vocab_size=hf_model.config.vocab_size
)
grammar_compiler = xgrammar.GrammarCompiler(
    tokenizer_info
)

compiled_grammar = grammar_compiler.compile_json_schema(json.dumps(schema))
logit_processor = xgrammar.contrib.hf.LogitsProcessor(compiled_grammar)

model_inputs = hf_tokenizer(prompt, return_tensors="pt")
output_ids = hf_model.generate(**model_inputs, logits_processor=[logit_processor], max_length=200)

generated_ids = output_ids[0][len(model_inputs.input_ids[0]) :]
output:str = hf_tokenizer.decode(generated_ids, skip_special_tokens=True)
```

</details>

### Outlines

<details>
<summary>View Code Example</summary>

Requires `outlines` library. Install it using:
```bash
pip install outlines   
```

```python
import outlines
model = model = outlines.models.transformers(
    model_name="meta-llama/Llama-3.2-1B-Instruct"
)
generator = outlines.generate.json(
                model, schema_object=json.dumps(schema)
            )
output:str = json.dumps(generator(prompt))
```

</details>

### Llamacpp

<details>
<summary>View Code Example</summary>

Requires `llama_cpp` library. Install it using:

```bash
pip install llama-cpp-python
```

```python
import llama_cpp
from llama_cpp.llama_grammar import LlamaGrammar

model = llama_cpp.Llama.from_pretrained(repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF", filename="*Q8_0.gguf")
compiled_grammar = LlamaGrammar.from_json_schema(json.dumps(schema))

response = model.create_chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                grammar=compiled_grammar
            )
output:str = response["choices"][0]["message"]["content"]
```

</details>

## Output Validation

We validate the schemas using the [jsonschema](https://pypi.org/project/jsonschema/) library, ensuring compliance with the [JSON Schema Draft 2012](https://json-schema.org/draft/2020-12) specification. Additionally, we enable the `format` validation with a few custom format checkers for enhanced validation.

To install the required library, run:
```bash
pip install jsonschema
```

Here’s an example of validating a structured output against a schema:
```python
import json
from validation import validate_enhanaced

output:str = '{"cuisine": "Italian", "location": "New York", "price_range": {"max_price": 30, "min_price": 10}}'

validate_enhanaced(json.loads(output), schema)
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
