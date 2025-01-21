#!/usr/bin/env python3
import subprocess
import os
import threading
from threading import Lock
import random
import sys
import glob
import json

output_path = "tmp/output/"
cmd = ["python3", "-m", "maskbench.runner"]
log_lock = Lock()


def run_cmd(file_list: list[str]):
    command = cmd + file_list
    log_entry = f"Running command: {' '.join(command)}\n"
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        log_entry += f"{result.stderr}{result.stdout}\nExit code: {result.returncode}\n"
    except Exception as e:
        log_entry += f"Error running command: {e}\n"
    append_to_log(log_entry)


def append_to_log(entry: str):
    log_file = os.path.join(output_path, "log.txt")
    with log_lock:
        with open(log_file, "a") as log:
            log.write(entry + "\n")


def missing_files(file_list: list[str]):
    r = [
        f
        for f in file_list
        if not os.path.exists(os.path.join(output_path, os.path.basename(f)))
    ]
    random.shuffle(r)
    return r


def process_files_in_threads(file_list: list[str], thread_count=40, chunk_size=100):
    """
    Processes a list of files using a specified number of threads, each handling a chunk of files.

    :param file_list: List of input file names.
    :param thread_count: Number of threads to use.
    :param chunk_size: Number of files each thread should handle in a single batch.
    """
    file_list_lock = Lock()

    file_list0 = file_list
    file_list = missing_files(file_list)

    print(f"Total files: {len(file_list)}")

    def worker():
        nonlocal file_list

        while True:
            files_chunk = []
            with file_list_lock:
                if not file_list:
                    # we're out of files, see what's left based on the filesystem
                    file_list = missing_files(file_list0)
                if not file_list:
                    return
                chunk = min(chunk_size, (len(file_list) // thread_count) + 20)
                files_chunk = file_list[:chunk]
                del file_list[:chunk]

            run_cmd(files_chunk)

            unprocessed_files = missing_files(files_chunk)

            num_total = 0
            with file_list_lock:
                file_list.extend(unprocessed_files)
                num_total = len(file_list)
                random.shuffle(file_list)

            n_unprocessed = len(unprocessed_files)
            n_processed = len(files_chunk) - n_unprocessed
            print(f"{n_processed} + {n_unprocessed}; {num_total} left.")

    threads = []
    for _ in range(thread_count):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    file_list = []
    if len(sys.argv) < 3:
        raise Exception(
            "Usage: python3 run_maskbench.py <--llg|--xgr|--outlines|...> <file> [<file>...]"
        )
    cmd.append(sys.argv[1])
    output_path = "tmp/out" + sys.argv[1]
    cmd.append("--output")
    cmd.append(output_path)
    for arg in sys.argv[2:]:
        if arg.endswith(".json"):
            file_list.append(arg)
        else:
            file_list.extend(glob.glob(arg + "/*.json"))

    if not file_list:
        raise Exception("No files found")

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "meta.txt"), "w") as meta:
        meta.write(json.dumps({"cmd": cmd}, indent=2))

    process_files_in_threads(file_list, thread_count=40, chunk_size=100)
