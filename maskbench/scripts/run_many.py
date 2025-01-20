#!/usr/bin/env python3
import subprocess
import os
import threading
from threading import Lock
import random
import sys
import glob

output_path = "tmp/output/"
cmd = ["python", "scripts/xgr/xgr_test.py"]


def check_file_outputs(file_list: list[str]):
    """
    Runs "python xgr_test.py f1 f2 ...", collects stderr and stdout,
    and checks which of the files in file_list created corresponding output files in output_path.

    :param file_list: List of input file names.
    :return: Dictionary with file names as keys and boolean values indicating if they created an output file.
    """

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
        log_entry += f"{result.stderr}{result.stdout}"
        append_to_log(log_entry)
    except Exception as e:
        log_entry += f"Error running command: {e}\n"
        append_to_log(log_entry)
        return {}

    # Check which files produced corresponding output files
    output_status = {}
    for file_name in file_list:
        output_file = os.path.join(output_path, os.path.basename(file_name))
        output_status[file_name] = os.path.exists(output_file)

    return output_status


def append_to_log(entry: str):
    """
    Atomically appends an entry to the log file 'tmp/log.txt'.

    :param entry: The log entry to append.
    """
    log_file = os.path.join(output_path, "log.txt")
    with Lock():
        with open(log_file, "a") as log:
            log.write(entry + "\n")


def process_files_in_threads(file_list: list[str], thread_count=40, chunk_size=100):
    """
    Processes a list of files using a specified number of threads, each handling a chunk of files.

    :param file_list: List of input file names.
    :param thread_count: Number of threads to use.
    :param chunk_size: Number of files each thread should handle in a single batch.
    """
    file_list_lock = Lock()

    file_list = [
        f
        for f in file_list
        if not os.path.exists(os.path.join(output_path, os.path.basename(f)))
    ]
    print(f"Total files: {len(file_list)}")

    random.shuffle(file_list)

    def worker():
        """Worker function to process chunks of files until the file list is empty."""
        while True:
            files_chunk = []
            with file_list_lock:
                if not file_list:
                    break
                chunk = min(chunk_size, (len(file_list) // thread_count) + 1)
                files_chunk = file_list[:chunk]
                del file_list[:chunk]

            results = check_file_outputs(files_chunk)

            unprocessed_files = []
            processed_files = []
            for file, status in results.items():
                if not status:
                    unprocessed_files.append(file)
                else:
                    processed_files.append(file)

            num_total = 0
            with file_list_lock:
                file_list.extend(unprocessed_files)
                num_total = len(file_list)
                random.shuffle(file_list)

            print(
                f"{len(processed_files)} + {len(unprocessed_files)}; {num_total} left."
            )

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
    cmd.append(sys.argv[1])
    output_path = "tmp/out" + sys.argv[1]
    cmd.append("--output")
    cmd.append(output_path)
    for arg in sys.argv[2:]:
        if arg.endswith(".json"):
            file_list.append(arg)
        else:
            file_list.extend(glob.glob(arg + "/*.json"))

    os.makedirs(output_path, exist_ok=True)
    process_files_in_threads(file_list, thread_count=40, chunk_size=100)
