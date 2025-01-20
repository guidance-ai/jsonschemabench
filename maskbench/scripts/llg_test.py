#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import time
import resource

from huggingface_hub import hf_hub_download

import llguidance as llg

output_path = "tmp/output/"


def time_us(prev: float) -> int:
    return int((time.monotonic() - prev) * 1000000)


def process_file(file: str):
    id = os.path.basename(file)
    output_name = output_path + id
    if os.path.exists(output_name):
        return None

    try:
        with open(output_name, "x") as f:
            f.write(json.dumps({"pending_file": 1}, indent=2))
    except FileExistsError:
        return None

    with open(file) as f:
        pos_data = json.loads(f.read())

    grammars = json.dumps({"grammars": [{"json_schema": pos_data["schema"]}]})

    all_mask_us = []
    status = {
        "id": id,
        "ttfm_us": 0,
        "max_ttfm_us": 0,
        "masks_us": 0,
        "max_mask_us": 0,
        "num_tokens": 0,
        "num_tests": len(pos_data["tests"]),
        "all_mask_us": all_mask_us,
        "num_valid_tests": 0,
        "num_invalid_tests": 0,
    }

    try:
        t0 = time.monotonic()
        interp = llg.LLInterpreter(
            tokenizer, grammars, enable_backtrack=False, enable_ff_tokens=False
        )
        interp.start_without_prompt()
    except Exception as e:
        status["compile_error"] = repr(e)
        with open(output_name, "w") as f:
            f.write(json.dumps(status, indent=2))
        return status

    status["ttfm_us"] = time_us(t0)
    status["max_ttfm_us"] = status["ttfm_us"]

    masks_us = 0
    max_mask_us = 0
    num_tokens = 0

    mask_data = bytearray((tokenizer.vocab_size + 7) // 8)

    for i, test in enumerate(pos_data["tests"]):
        matcher = interp.deep_copy()
        instance = json.dumps(test["data"], indent=None)
        tokens = tokenizer.tokenize_str(instance)

        t1 = time.monotonic()
        accepted = True
        for tidx, t in enumerate(tokens):
            t2 = time.monotonic()
            res = matcher.compute_mask_into(mask_data)
            ok = (mask_data[t // 8] & (1 << (t % 8))) != 0
            if ok:
                matcher.commit_token(t)
            mask_time = time_us(t2)
            # print(f"Token {tidx} {repr(tokenizer.decode([t]))}: {ok}", file=sys.stderr)
            num_tokens += 1
            masks_us += mask_time
            all_mask_us.append(mask_time)
            if mask_time > max_mask_us:
                max_mask_us = mask_time
            if not ok:
                accepted = False
                break

        if accepted and not test["valid"]:
            status["validation_error"] = f"test #{i}: should reject but didn't"
        elif not accepted and test["valid"]:
            status["validation_error"] = f"test #{i}: should accept but didn't"
        else:
            if test["valid"]:
                status["num_valid_tests"] += 1
            else:
                status["num_invalid_tests"] += 1

    status["masks_us"] = masks_us
    status["max_mask_us"] = max_mask_us
    status["num_tokens"] = num_tokens

    with open(output_name, "w") as f:
        f.write(json.dumps(status, indent=2))
    return status


def main():
    global tokenizer

    limit_gb = 32
    limit_bytes = limit_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    # Get tokenizer info
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer_json_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
    with open(tokenizer_json_path, "r") as f:
        tokenizer = llg.LLTokenizer(f.read())

    files = []
    for arg in sys.argv[1:]:
        if arg.endswith(".json"):
            files.append(arg)
        else:
            files.extend(glob.glob(arg + "/*.json"))
    # print(len(files), file=sys.stderr)
    random.shuffle(files)

    os.makedirs(output_path, exist_ok=True)

    for f in files:
        # print(f, file=sys.stderr)
        process_file(f)


main()
