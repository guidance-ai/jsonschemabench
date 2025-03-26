# based on https://github.com/mlc-ai/xgrammar/blob/172457f/examples/benchmark/cibench_grammar_compile_mask_gen.py
# by @zanderjiang

# The json schema conversion has been improved to take more types, and the data is validated now

import json
import os
from typing import Any, Dict, List, Tuple
import requests
import jsonschema
from jsonschema import Draft202012Validator


def download_gorilla_file(filename: str) -> Tuple[List, List]:
    base_url = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/data"
    function_url = f"{base_url}/{filename}"
    answer_url = f"{base_url}/possible_answer/{filename}"

    print(f"Downloading {filename} from GitHub...")

    try:
        function_response = requests.get(function_url)
        function_response.raise_for_status()
        function_text = function_response.text

        functions_data = []
        for line in function_text.strip().split("\n"):
            if line.strip():
                try:
                    functions_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing function line in {filename}: {e}")

        answer_response = requests.get(answer_url)
        answer_response.raise_for_status()
        answer_text = answer_response.text

        answers_data = []
        for line in answer_text.strip().split("\n"):
            if line.strip():
                try:
                    answers_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing answer line in {filename}: {e}")

        print(
            f"Successfully downloaded {filename}: {len(functions_data)} functions, {len(answers_data)} answers"
        )
        return functions_data, answers_data
    except requests.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return [], []


def load_gorilla_data() -> List[Dict[str, Any]]:
    gorilla_data = []

    # excluding live test cases part of BFCL v2/v3
    file_patterns = [
        "BFCL_v3_java.json",
        "BFCL_v3_javascript.json",
        "BFCL_v3_multiple.json",
        "BFCL_v3_parallel.json",
        "BFCL_v3_parallel_multiple.json",
        "BFCL_v3_simple.json",
        "BFCL_v3_sql_unused.json",
    ]

    filtered_count = 0

    all_funs = []
    all_data = []

    for filename in file_patterns:

        cache_file = "all_" + filename

        if not os.path.exists(cache_file):
            functions_data, answers_data = download_gorilla_file(filename)
            if not functions_data or not answers_data:
                print(f"Skipping {filename} - failed to download data")
                continue
            with open(cache_file, "w") as f:
                json.dump([functions_data, answers_data], f, indent=2)
            print(f"Downloaded {filename} and saved to {cache_file}")
        else:
            with open(cache_file, "r") as f:
                functions_data, answers_data = json.load(f)

        all_funs.extend(json.loads(json.dumps(functions_data)))
        all_data.extend(json.loads(json.dumps(answers_data)))

        if not functions_data or not answers_data:
            print(f"Skipping {filename} - failed to download data")
            continue

        # with open("all_" + filename, "w") as f:
        #     json.dump([functions_data, answers_data], f)

        print(f"Processing {filename}...")

        validation_errors = []

        answers_by_id = {item["id"]: item for item in answers_data}

        for item in functions_data:
            item_id = item["id"]

            if item_id not in answers_by_id:
                print(f"Warning: No answer found for item {item_id}")
                continue

            if "function" not in item or not item["function"]:
                print(f"Warning: No function definition for item {item_id}")
                filtered_count += 1
                continue

            # if len(item["function"]) > 1:
            #     print(f"Skipping item {item_id} - contains multiple functions ({len(item['function'])})")
            #     filtered_count += 1
            #     continue

            orig = item["function"]
            functions = [convert_function_to_schema(f) for f in orig]

            if len(functions) == 1:
                schema = functions[0]
            else:
                schema = {"anyOf": functions}

            answer = answers_by_id[item_id]
            if "ground_truth" not in answer or not answer["ground_truth"]:
                print(f"Warning: No ground truth for item {item_id}")
                filtered_count += 1
                continue

            ground_truth = answer["ground_truth"][
                0]  # Use the first ground truth

            completion = convert_ground_truth_to_completion(ground_truth)

            try:
                jsonschema.validate(
                    instance=completion,
                    schema=schema,
                    format_checker=Draft202012Validator.FORMAT_CHECKER)
            except jsonschema.ValidationError as e:
                msg = f"Validation error for item {item_id}: {e}"
                # print(msg + "\n\n")
                validation_errors.append(msg)
                filtered_count += 1
                continue

            gorilla_data.append({
                "id": item_id,
                "source": filename,
                "schema_orig": orig,
                "schema": schema,
                "completion": completion,
            })

            with open("out/BFCL_" + item_id + ".json", "w") as f:
                json.dump(
                    {
                        "description":
                        f"{filename} {item_id}",
                        "schema":
                        schema,
                        "tests": [{
                            "description": "from BFCL ground truth",
                            "valid": True,
                            "data": completion,
                        }]
                    },
                    f,
                    indent=4)

    with open("all.json", "w") as f:
        json.dump([all_funs, all_data], f, indent=2)
    with open("combined.json", "w") as f:
        json.dump(gorilla_data, f, indent=2)

    print(f"Validation errors: {len(validation_errors)}")
    print(
        f"Loaded {len(gorilla_data)} examples from Gorilla BFCL dataset (filtered out {filtered_count} examples)"
    )
    return gorilla_data


def convert_one(value):
    param_type = value.get("type", "string").lower()
    del value["type"]

    if param_type in ("integer", "long"):
        schema_def = {"type": "integer"}
        for k in ["minimum", "maximum"]:
            if k in value:
                schema_def[k] = value[k]
                del value[k]
    elif param_type in ("float", "number", "double"):
        schema_def = {"type": "number"}
        for k in ["minimum", "maximum"]:
            if k in value:
                schema_def[k] = value[k]
                del value[k]
    elif param_type == "boolean":
        schema_def = {"type": "boolean"}
    elif param_type in ("hashmap", "map", "dict", "dictionary"):
        if "properties" in value:
            props: dict = value["properties"]
            del value["properties"]
            req = list(props.keys())
            if "required" in value:
                req = value["required"]
                del value["required"]
            schema_def = {
                "type": "object",
                "properties": {},
                "required": req,
                "additionalProperties": False
            }
            for k, v in props.items():
                schema_def["properties"][k] = convert_one(v)
        else:
            schema_def = {"type": "object", "additionalProperties": True}
    elif param_type in ("array", "list", "arraylist", "tuple"):
        if "items" in value:
            items = value["items"]
            del value["items"]
            schema_def = {"type": "array", "items": convert_one(items)}
        else:
            print(value)
            schema_def = {"type": "array", "items": {"type": "string"}}
    elif param_type == "any":
        schema_def = {}  # No type restriction
    elif param_type in ("string", "char"):
        schema_def = {"type": "string"}
        for k in ["enum", "format"]:
            if k in value:
                schema_def[k] = value[k]
                del value[k]
    else:
        print("unknown: " + param_type)
        schema_def = {"type": "string"}

    for k in ["description", "default", "example", "optional"]:
        if k in value:
            del value[k]
    if value:
        print(value)

    return schema_def


def convert_function_to_schema(function_def: Dict) -> dict:
    """Convert a Gorilla function definition to a JSON schema string with improved type handling."""
    function_name = function_def["name"]
    parameters = function_def["parameters"]

    parameters = json.loads(json.dumps(parameters))

    schema = {
        "type": "object",
        "properties": {
            function_name: {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        },
        "required": [function_name],
        "additionalProperties": False
    }

    for key, value in parameters.get("properties", {}).items():
        schema_def = convert_one(value)
        schema["properties"][function_name]["properties"][key] = schema_def

    required_fields = parameters.get("required", [])
    if required_fields:
        schema["properties"][function_name]["required"] = required_fields

    return schema


def convert_ground_truth_to_completion(ground_truth: Dict) -> dict:
    """Convert a Gorilla ground truth to a completion string with improved handling of nested structures."""
    function_name = list(ground_truth.keys())[0]
    params = ground_truth[function_name]

    transformed_params = {}
    for key, values in params.items():
        if isinstance(values, list) and len(values) == 1 and isinstance(
                values[0], dict):
            nested_obj = {}

            for nested_key, nested_values in values[0].items():
                if isinstance(nested_values, list) and nested_values:
                    nested_obj[nested_key] = nested_values[0]
                else:
                    nested_obj[nested_key] = nested_values

            transformed_params[key] = nested_obj
        elif isinstance(values, list) and values:
            transformed_params[key] = values[0]
        else:
            transformed_params[key] = None

    completion = {function_name: transformed_params}

    return completion


if __name__ == "__main__":
    gorilla_data = load_gorilla_data()
