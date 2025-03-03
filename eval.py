import torch
import re
import os
import random
import json
import yaml
from pathlib import Path
import transformers
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from jsonschema import ValidationError

import argparse
import random
import os
import numpy as np
from validation import validate_enhanaced

transformers.logging.set_verbosity(40)
MAX_NEW_TOKENS = 500
INFER_FILE_NUM = 200


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/dlabscratch1/cjin/data",
        help="The root folder of the data.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="./log",
        help="The root folder of the data.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Glaiveai2K",
        help="The dataset name",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/glaiveai_json_schema_gen_2shot_chat.yaml",
        help="The config path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--validation_result_file",
        type=str,
        default="validation_results.json",
        help="The file recording the validation result"
    )

    args = parser.parse_args()
    return args



def load_model_tokenizer(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    # print('tokenizer chat template: ', tokenizer.chat_template)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer



def extract_json_schema(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data



def load_dataset(dataset_path):
    dataset_dir = Path(dataset_path)
    schemas = {}

    for file_path in dataset_dir.glob("*.json"):
        schema = extract_json_schema(file_path)
        schemas[file_path.name] = schema
    
    return schemas



def load_yaml_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)  # 解析 YAML
    return config



def process_system_message(config):
    prompter = config.get("prompter", {})

    system_message = prompter.get("system_message", "")
    demo_pool = prompter.get("demo", {}).get("demo_pool", [])
    num_demo = prompter.get("demo", {}).get("num_demo", 2)
    demo_selection = prompter.get("demo", {}).get("demo_selection", "first")
    demo_separator = prompter.get("demo", {}).get("demo_separator", "\n\n")
    
    system_message += "\nExamples:\n"
    for i in range(num_demo):
        demo = demo_pool[i]
        system_message += f"Example {i + 1}:\n"
        system_message = system_message + "Input: " + demo["input"] + "\n"
        system_message = system_message + "Output: " + demo["output"] + "\n" 
    
    return system_message


def generate_output(model, tokenizer, system_message, user_message):
    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user", 
            "content": user_message,
        },
    ]
    tokenized_messages = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )
    
    outputs = model.generate(tokenized_messages, max_new_tokens=MAX_NEW_TOKENS)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    input_text = tokenizer.decode(tokenized_messages[0], skip_special_tokens=True)
    output_text = generated_text[len(input_text):].strip()
    
    return output_text



def validation_process(model, tokenizer, dataset_path, system_message, log_file_dir):
    validation_results = defaultdict(dict)
    
    json_files = list(dataset_path.glob("*.json"))[:INFER_FILE_NUM]  # all json files
    total_files = len(json_files) 

    with tqdm(total=total_files, desc="Processing JSON files", unit="file") as pbar:
        for file_path in json_files:
            schema = extract_json_schema(file_path)
            user_message = json.dumps(schema, indent=4)
            
            # print("system_message: ", system_message)
            # print("user message: ", user_message)
            
            output_text = generate_output(
                model=model, 
                tokenizer=tokenizer, 
                system_message=system_message, 
                user_message=user_message
            )
            # print(output_text)
            
            try:
                parsed_output = json.loads(output_text)
            except json.JSONDecodeError as e:
                is_valid = False
                validation_message = f"Invalid JSON format: {str(e)}"
                parsed_output = None 
            else:
                # JSON Schema check
                try:
                    is_valid = validate_enhanaced(parsed_output, schema)
                except ValidationError as e:
                    is_valid = False
                    validation_message = str(e)
                else:
                    validation_message = "Valid JSON"

            # write json when parsed_output is json
            if parsed_output is not None:
                json_log_file = os.path.join(log_file_dir, f"{file_path.stem}_output.json")
                with open(json_log_file, "w", encoding="utf-8") as f:
                    json.dump(parsed_output, f, indent=4, ensure_ascii=False)
            # write txt when parsed_output is not json
            else:
                raw_log_file = os.path.join(log_file_dir, f"{file_path.stem}_output.txt")
                with open(raw_log_file, "w", encoding="utf-8") as f:
                    f.write(output_text)
                
            validation_results[file_path.stem] = {
                "is_valid": is_valid,
                "message": validation_message
            }
            pbar.update(1)
            
    return validation_results



def calculate_accuracy(validation_results):
    valid_count = sum(1 for result in validation_results.values() if result["is_valid"])
    total_count = len(validation_results)
    accuracy = (valid_count / total_count) * 100 if total_count > 0 else 0
    return accuracy



def main():
    args = parse_args()
    seed_everything(args.seed)
    config = load_yaml_config(args.config_path)
    model, tokenizer = load_model_tokenizer(args.model_name_or_path)

    dataset_path = Path(os.path.join(args.data_root, args.dataset))
    system_message = process_system_message(config)
    log_file_dir = os.path.join(args.log_root, args.dataset)

    validation_results = validation_process(model, tokenizer, dataset_path, system_message, log_file_dir)
    accuracy = calculate_accuracy(validation_results)
    print(f"Validation Accuracy: {accuracy:.2f}%")
    
    validation_result_file_path = os.path.join(log_file_dir, args.validation_result_file)
    with open(validation_result_file_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=4, ensure_ascii=False)
    
    
if __name__ == "__main__":
    main()