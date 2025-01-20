#!/usr/bin/env python3

import sys
import json
import glob
import os
import random
import time
import resource
import argparse

import llguidance as llg

output_path = "tmp/output/"


def time_us(prev: float) -> int:
    return int((time.monotonic() - prev) * 1000000)


def build_matcher(compiled_grammar):
    import xgrammar as xgr

    return xgr.GrammarMatcher(compiled_grammar)


allocate_bitmask = None


def process_file(file: str):
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

    if llg_tokenizer:
        mask_data = bytearray(bytearray((llg_tokenizer.vocab_size + 7) // 8))
    elif outlines_tokenizer:
        pass
    else:
        token_bitmask = allocate_bitmask()

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
        if llg_tokenizer:
            grammars = json.dumps({"grammars": [{"json_schema": pos_data["schema"]}]})
            interp = llg.LLInterpreter(
                llg_tokenizer, grammars, enable_backtrack=False, enable_ff_tokens=False
            )
            interp.start_without_prompt()
        elif outlines_tokenizer:
            from outlines_core.fsm.json_schema import build_regex_from_schema
            from outlines_core.fsm.guide import Write, Generate
            from outlines.fsm.guide import RegexGuide

            rx = build_regex_from_schema(json.dumps(pos_data["schema"]))
            guide = RegexGuide.from_regex(rx, outlines_tokenizer)

            def allows_token(inst, t):
                if isinstance(inst, Write):
                    return t in inst.tokens
                elif isinstance(inst, Generate) and inst.tokens is not None:
                    return t in inst.tokens
                else:
                    return False

        else:
            schema = json.dumps(pos_data["schema"])
            compiled_grammar = xgr_compiler.compile_json_schema(
                schema, any_whitespace=xgr_any_whitespace, strict_mode=True
            )
            # print(compiled_grammar.grammar, file=sys.stderr)
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
        if llg_tokenizer:
            llg_matcher = interp.deep_copy()
        elif outlines_tokenizer:
            guide_state = guide.initial_state
        else:
            matcher = build_matcher(compiled_grammar)

        instance = json.dumps(test["data"], indent=None, ensure_ascii=False)
        if llg_tokenizer:
            tokens = llg_tokenizer.tokenize_str(instance)
        else:
            tokens = xgr_tokenizer.encode(instance, add_special_tokens=False)

        accepted = True
        for tidx, t in enumerate(tokens):
            t2 = time.monotonic()
            if llg_tokenizer:
                res = llg_matcher.compute_mask_into(mask_data)
                ok = (mask_data[t // 8] & (1 << (t % 8))) != 0
                if ok:
                    llg_matcher.commit_token(t)
            elif outlines_tokenizer:
                inst = guide.get_next_instruction(guide_state)
                ok = allows_token(inst, t)
                if ok:
                    guide_state = guide.get_next_state(guide_state, t)
            else:
                matcher.fill_next_token_bitmask(token_bitmask)
                ok = matcher.accept_token(t)
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
    global xgr_compiler, xgr_tokenizer, llg_tokenizer
    global xgr_strict, xgr_any_whitespace
    global outlines_tokenizer
    global output_path

    limit_gb = 40
    limit_bytes = limit_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

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

    if args.output:
        output_path = args.output

    # Get tokenizer info
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    outlines_tokenizer = None
    llg_tokenizer = None

    if args.xgr or args.xgr_compliant:
        from transformers import AutoTokenizer, AutoConfig
        import xgrammar as xgr

        xgr_tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        # This can be larger than tokenizer.vocab_size due to paddings
        full_vocab_size = config.vocab_size
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            xgr_tokenizer, vocab_size=full_vocab_size
        )
        global allocate_bitmask
        allocate_bitmask = lambda: xgr.allocate_token_bitmask(
            1, tokenizer_info.vocab_size
        )
        xgr_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1)
        if args.xgr_compliant:
            xgr_strict = False
            xgr_any_whitespace = True
        else:
            xgr_strict = True
            xgr_any_whitespace = False
    elif args.llg:
        from huggingface_hub import hf_hub_download

        tokenizer_json_path = hf_hub_download(
            repo_id=model_id, filename="tokenizer.json"
        )
        with open(tokenizer_json_path, "r") as f:
            llg_tokenizer = llg.LLTokenizer(f.read())
    elif args.outlines:
        from transformers import AutoTokenizer
        from outlines.models.transformers import TransformerTokenizer

        xgr_tokenizer = AutoTokenizer.from_pretrained(model_id)
        outlines_tokenizer = TransformerTokenizer(xgr_tokenizer)
    else:
        raise Exception("No mode specified")

    files = []
    for arg in sys.argv[1:]:
        if arg.endswith(".json"):
            files.append(arg)
        else:
            files.extend(glob.glob(arg + "/*.json"))
    print(len(files), file=sys.stderr)
    random.shuffle(files)

    os.makedirs(output_path, exist_ok=True)

    for f in files:
        process_file(f)


main()
