#!/usr/bin/env python3
import subprocess
import os
import threading
from threading import Lock
import random
import sys
import glob
import json
import time

output_path = "tmp/output/"
cmd = ["python3", "-m", "maskbench.runner", "--multi"]
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
    num_processed = 0
    num_all_files = len(file_list)

    t0 = time.monotonic()

    print(f"Total files: {num_all_files}")

    if not file_list:
        print("All files processed.")
        return

    thread_count = min(thread_count, len(file_list))

    def worker():
        nonlocal file_list, num_processed

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

            processed_here = len(files_chunk) - len(unprocessed_files)

            num_total = 0
            with file_list_lock:
                num_processed += processed_here
                file_list.extend(unprocessed_files)
                num_total = len(file_list)
                random.shuffle(file_list)

            n_unprocessed = len(unprocessed_files)
            n_processed = len(files_chunk) - n_unprocessed
            now = time.monotonic() - t0
            perc_done = num_processed / num_all_files * 100
            est_time_left = (now / (1 + num_processed) * num_all_files) - now
            print(
                f"{now:.1f}s {n_processed} + {n_unprocessed}; {num_total} pending; {perc_done:.1f}% done, est. {est_time_left:.1f}s left"
            )

    threads = []
    for _ in range(thread_count):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print(f"Total time: {time.monotonic() - t0:.2f}s")


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from maskbench.runner import setup_argparse, get_output, get_files, get_engine

    parser = setup_argparse()
    args = parser.parse_args()

    engine = get_engine(args)
    output_path = get_output(args)

    file_list = get_files(args)
    if not file_list:
        raise Exception("No files found")

    args_to_pass = [a for a in sys.argv[1:] if a not in args.files]
    cmd += args_to_pass

    info = f"{len(file_list)} files, timeout {args.time_limit}s, memory {args.mem_limit}GB, "
    info += f"output {output_path}; {args.num_threads} threads; cmd: {' '.join(cmd)}"

    print(info, file=sys.stderr)

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "meta.txt"), "w") as meta:
        meta.write(
            json.dumps(
                {
                    "id": engine.get_id(),
                    "name": engine.get_name(),
                    "module": engine.get_module(),
                    "module_version": engine.get_version(),
                    "cmd": cmd,
                    "info": info,
                },
                indent=2,
            )
        )

    process_files_in_threads(file_list, thread_count=args.num_threads, chunk_size=100)
