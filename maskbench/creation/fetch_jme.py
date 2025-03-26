import json
import os
import jsonschema
from jsonschema import Draft202012Validator
import datasets

def main():
    dataset = datasets.load_dataset("NousResearch/json-mode-eval", split="train")
    valid_count = 0
    filtered_count = 0
    validation_errors = []

    for idx, item in enumerate(dataset):
        print(item.keys())
        schema = json.loads( item["schema"] )
        completion = json.loads( item["completion"] )

        try:
            jsonschema.validate(instance=completion, schema=schema, 
                                format_checker=Draft202012Validator.FORMAT_CHECKER)
        except jsonschema.ValidationError as e:
            print(f"Validation error for item {idx}: {e}")
            validation_errors.append(f"Item {idx}: {e}")
            filtered_count += 1
            continue

        valid_count += 1
        output_data = {
            "description": f"NousResearch/json-mode-eval {idx}",
            "schema": schema,
            "tests": [{
                "description": "from NousResearch/json-mode-eval dataset",
                "valid": True,
                "data": completion
            }]
        }
        with open(f"data/JME_{idx}.json", "w") as f:
            json.dump(output_data, f, indent=4)

    print(f"Validation errors: {len(validation_errors)}")
    print(f"Loaded {valid_count} examples from json-mode-eval dataset (filtered out {filtered_count} examples)")

if __name__ == "__main__":
    main()
