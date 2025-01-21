#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import time
import resource
import argparse
import signal

from .engine import Engine
from transformers import AutoTokenizer


def time_us(prev: float) -> int:
    return int((time.monotonic() - prev) * 1000000)


def process_file(engine: Engine, file: str):
    id = os.path.basename(file)
    output_name = os.path.join(output_path, id)
    if os.path.exists(output_name):
        return None

    try:
        with open(output_name, "x") as f:
            f.write(json.dumps({"pending_file": 1}, indent=2))
    except FileExistsError:
        return None

    print(file, file=sys.stderr)

    with open(file) as f:
        pos_data = json.loads(f.read())

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
        engine.compile_grammar(pos_data["schema"])
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

    for i, test in enumerate(pos_data["tests"]):
        engine.reset()

        instance = json.dumps(test["data"], indent=None, ensure_ascii=False)
        tokens = engine.tokenizer.encode(instance, add_special_tokens=False)

        accepted = True
        for tidx, t in enumerate(tokens):
            t2 = time.monotonic()
            engine.compute_mask()
            ok = engine.commit_token(t)
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
    global output_path

    limit_gb = 40
    limit_bytes = limit_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    time_limit_s = 15 * 60

    parser = argparse.ArgumentParser(description="Run mask computation.")
    parser.add_argument("--xgr", action="store_true", help="Enable XGrammar")
    parser.add_argument(
        "--xgr-compliant",
        action="store_true",
        help="Enable XGrammar in compliant (non-strict, any whitespace) mode",
    )
    parser.add_argument("--llg", action="store_true", help="Enable LLGuidance")
    parser.add_argument("--outlines", action="store_true", help="Enable Outlines")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument(
        "files", metavar="file", type=str, nargs="+", help="List of files to process"
    )

    args = parser.parse_args()

    # Get tokenizer info
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    if args.xgr or args.xgr_compliant:
        from .xgr_engine import XgrEngine

        engine = XgrEngine()
        engine.compliant = args.xgr_compliant
        id = "xgr_compliant" if args.xgr_compliant else "xgr"
    elif args.llg:
        from .llg_engine import LlgEngine

        engine = LlgEngine()
        id = "llg"
    elif args.outlines:
        from .outlines_engine import OutlinesEngine

        engine = OutlinesEngine()
        id = "outlines"
    else:
        raise Exception("No mode specified")

    if args.output:
        output_path = args.output
    else:
        output_path = f"tmp/out-{id}"

    engine.tokenizer_model_id = model_id
    engine.tokenizer = AutoTokenizer.from_pretrained(model_id)  # type: ignore

    engine.init()

    files = []
    for arg in sys.argv[1:]:
        if arg.endswith(".json"):
            files.append(arg)
        else:
            files.extend(glob.glob(arg + "/*.json"))
    print(f"{len(files)} files, timeout {time_limit_s}s", file=sys.stderr)
    random.shuffle(files)

    os.makedirs(output_path, exist_ok=True)

    # rely on default SIGALRM handler (terminates the process)
    # def timeout_handler(signum, frame):
    #     print(f"Timeout ({time_limit_s}s). Terminating...", file=sys.stderr)
    #     os.kill(os.getpid(), signal.SIGKILL)

    # signal.signal(signal.SIGALRM, timeout_handler)

    for f in files:
        signal.alarm(time_limit_s)
        process_file(engine, f)
        signal.alarm(0)


if __name__ == "__main__":
    main()
