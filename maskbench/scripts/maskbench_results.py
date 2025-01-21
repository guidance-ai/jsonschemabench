#!/usr/bin/env python3

import json
import math
import glob
import sys
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import math


class Stats:
    def __init__(self) -> None:
        self.meta = {}
        self.ttfm_us = 0
        self.max_ttfm_us = 0
        self.masks_us = 0
        self.avg_mask_us = 0
        self.masks_us_under_10ms = 0
        self.num_masks_under_10ms = 0
        self.avg_masks_under_10ms = 0
        self.masks_us_over_10ms = 0
        self.num_masks_over_10ms = 0
        self.avg_masks_over_10ms = 0
        self.max_mask_us = 0
        self.num_tokens = 0
        self.num_schemas = 0
        self.num_crashes_or_timeouts = 0
        self.num_timeouts = 0
        self.num_segv = 0
        self.num_abort = 0
        self.num_schemas_ok = 0
        self.num_compilation_errors = 0
        self.num_validation_errors = 0
        self.num_invalidation_errors = 0
        self.num_tests = 0
        self.num_valid_tests = 0
        self.num_invalid_tests = 0

    def load_json(self, data: dict):
        for k in self.__dict__.keys():
            if k in data:
                self.__dict__[k] = data[k]


def log_fraction_plot(times: list[int]):
    times.sort()
    cutoff = 1
    mult = 1.3
    count = 0
    csv = "cutoff time,frac left\n"
    total = len(times)
    for t in times:
        while t > cutoff:
            csv += f"{cutoff/1000.0},{(total - count)/total}\n"
            cutoff = int(cutoff * mult) + 1
        count += 1
    return csv


def histogram_position(us: int):
    return int(math.floor(math.log10(max(1, us - 1))))


def us_to_str(us: int):
    if us < 1000:
        return f"{us}us"
    if us < 1000000:
        return f"{us//1000}ms"
    return f"{us//1000000}s"


def read_json(filename: str):
    with open(filename) as f:
        data = json.load(f)
    return data


def main(folder: str):
    # for llg rust: "tmp/llg_results.json"
    if not os.path.isdir(folder):
        raise Exception(f"Not a directory: {folder}")
    files = glob.glob(folder + "/*.json")
    files = sorted(files)

    stats = Stats()
    stats.meta = read_json(folder + "/meta.txt")
    ttfm_us = []
    all_masks_us = []
    histogram_us = [0] * 10
    histogram_num = [0] * 10
    for f in files:
        with open(f) as f:
            data = json.load(f)
        elts = [data]
        if isinstance(data, list):
            elts = data
        for data in elts:
            stats.num_schemas += 1
            if "num_tests" not in data:
                stats.num_crashes_or_timeouts += 1
                continue
            stats.num_tests += data["num_tests"]
            if "compile_error" in data:
                stats.num_compilation_errors += 1
            else:
                stats.ttfm_us += data["ttfm_us"]
                ttfm_us.append(data["ttfm_us"])
                stats.max_ttfm_us = max(data["max_ttfm_us"], stats.max_ttfm_us)
                if "masks_us" in data:
                    stats.masks_us += data["masks_us"]
                    stats.max_mask_us = max(data["max_mask_us"], stats.max_mask_us)
                    stats.num_tokens += data["num_tokens"]
                if "validation_error" in data:
                    if "should reject" in data["validation_error"]:
                        stats.num_invalidation_errors += 1
                    else:
                        stats.num_validation_errors += 1
                else:
                    stats.num_schemas_ok += 1
                stats.num_valid_tests += data["num_valid_tests"]
                stats.num_invalid_tests += data["num_invalid_tests"]
                all_masks_us.extend(data["all_mask_us"])
                for us in data["all_mask_us"]:
                    p = histogram_position(us)
                    histogram_us[p] += us
                    histogram_num[p] += 1
                    if us < 10000:
                        stats.masks_us_under_10ms += us
                        stats.num_masks_under_10ms += 1
                    else:
                        stats.masks_us_over_10ms += us
                        stats.num_masks_over_10ms += 1
    stats.avg_masks_under_10ms = stats.masks_us_under_10ms // stats.num_masks_under_10ms
    stats.avg_masks_over_10ms = stats.masks_us_over_10ms // stats.num_masks_over_10ms
    stats.avg_mask_us = stats.masks_us // stats.num_tokens
    print(json.dumps(stats.__dict__, indent=2))
    with open(folder + "/stats.txt", "w") as f:
        f.write(json.dumps(stats.__dict__, indent=2))
    with open(folder + "/ttfm_us.csv", "w") as f:
        f.write(log_fraction_plot(ttfm_us))
    with open(folder + "/masks_us.csv", "w") as f:
        f.write(log_fraction_plot(all_masks_us))

    with open(folder + "/log.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "Exit code: -11":
                stats.num_segv += 1
            elif line == "Exit code: -6":
                stats.num_abort += 1
            elif line == "Exit code: -14":
                stats.num_timeouts += 1

    ps = [25, 50, 75, 90, 95, 99, 99.9, 100]

    def get_p(arr: list[int], p: float):
        idx = int(len(arr) * p / 100)
        if idx >= len(arr):
            idx = len(arr) - 1
        return arr[idx]

    entries = {}
    all_masks_us.sort()
    entries["TBM avg (us)"] = sum(all_masks_us) // len(all_masks_us)
    for p in ps:
        entries[f"TBM p{p}"] = get_p(all_masks_us, p)
    ttfm_us.sort()
    entries["TTFM avg (us)"] = sum(ttfm_us) // len(ttfm_us)
    for p in ps:
        entries[f"TTFM p{p}"] = get_p(ttfm_us, p)
    entries["tokens"] = stats.num_tokens
    entries["schemas"] = stats.num_schemas
    entries["passing"] = stats.num_schemas_ok
    entries["crashes"] = stats.num_crashes_or_timeouts
    entries["segv"] = stats.num_segv
    entries["oom"] = stats.num_abort
    entries["timeouts"] = stats.num_timeouts
    entries["compile errors"] = stats.num_compilation_errors
    entries["validation errors"] = stats.num_validation_errors
    entries["invalidation errors"] = stats.num_invalidation_errors
    print(json.dumps(entries, indent=2))
    with open(folder + "/entries.txt", "w") as f:
        f.write(json.dumps(entries, indent=2))
    # print(f"{'ttfm_p' + str(p):10}, {ttfm_us[int(len(ttfm_us) * p / 100)]}")

    num_masks = sum(histogram_num)
    h_csv = "above us,frac\n"
    for i in range(10)[1:]:
        frac = sum(histogram_num[i:]) * 100 / num_masks
        h_csv += f"{us_to_str(10**i):10}"
        h_csv += f","
        h_csv += f"{frac:1.15}"
        h_csv += f"\n"
    with open(folder + "/histogram.csv", "w") as f:
        f.write(h_csv)
    # print(h_csv)

    return stats, entries


def print_markdown_table(data):
    # Determine the column widths
    col_widths = [max(len(row[col]) for row in data) for col in range(len(data[0]))]

    # Generate the Markdown table
    def format_row(row):
        return (
            "| "
            + " | ".join(
                f"{cell:<{col_widths[i]}}" if i == 0 else f"{cell:>{col_widths[i]}}"
                for i, cell in enumerate(row)
            )
            + " |\n"
        )

    r = format_row(data[0])
    r += (
        "|"
        + "|".join(
            ":" + "-" * (width + 1) if i == 0 else "-" * (width + 1) + ":"
            for i, width in enumerate(col_widths)
        )
        + "|\n"
    )

    # Print the rows
    for row in data[1:]:
        r += format_row(row)

    print(r)
    return r


def format_time(time_us: int):
    if time_us < 1_000:
        result = f"{time_us}μs"
    elif time_us < 1_000_000:
        time_ms = time_us / 1_000
        if time_ms < 10:
            result = f"{time_ms:.1f}ms"
        else:
            result = f"{time_ms:.0f}ms"
    else:
        time_s = time_us / 1_000_000
        if time_s < 10:
            result = f"{time_s:.1f}s"
        else:
            result = f"{time_s:.0f}s"
    return result


def plot_metrics(data_list: list[dict], prefix: str, title: str):
    tbm_keys = [key for key in data_list[0].keys() if key.startswith(prefix)]
    labels = [data["name"] for data in data_list]
    values = {
        label: [data[key] for key in tbm_keys] for label, data in zip(labels, data_list)
    }

    # Calculate positions
    y = np.arange(len(tbm_keys))
    height = 0.8 / len(labels)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    bar_positions = []

    custom_colors = [
        "#000000",
        "#0379b6",
        "#039eb6",
        "#d16626",
    ]

    for i, (label, tbm_values) in enumerate(values.items()):
        pos = y + (len(labels) - i - 1) * height - 0.4 + (height / 2)
        bar_positions.append(pos)
        bars = ax.barh(
            pos,
            tbm_values,
            height,
            label=label,
            color=custom_colors[i % len(custom_colors)],
        )

        # Add annotations on the bars
        for j, (bar, value) in enumerate(zip(bars, tbm_values)):
            label_text = format_time(value)
            ax.text(
                bar.get_width() * 1.1,
                bar.get_y() + bar.get_height() / 2,
                label_text,
                ha="left",
                va="center",
                fontsize=8,
                rotation=0,
            )

            # Add speedup/slowdown factor
            all_values_for_key = [values[engine][j] for engine in labels]
            fastest = min(all_values_for_key)
            factor = value / fastest

            speedup_text = f"{factor:.0f}x"
            ax.text(
                bar.get_width() / 1.2,
                bar.get_y() + bar.get_height() / 2,
                speedup_text,
                ha="right",
                va="center",
                fontsize=8,
                rotation=0,
                color="white",
            )

    # Configure axes
    ax.set_xscale("log", base=10)
    ax.set_yticks(y)
    ax.set_yticklabels(tbm_keys, rotation=45, ha="right")
    ax.set_xlabel("Time (log scale, microseconds)")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"plots/{prefix.strip().lower()}.png", dpi=300)


if __name__ == "__main__":
    folders = sys.argv[1:]
    quick = False
    if len(folders) > 0 and folders[0] == "-q":
        quick = True
        folders = folders[1:]

    if not folders:
        metas = glob.glob("tmp/out--*/meta.txt")
        folders = [os.path.dirname(m) for m in metas]

    ents: list[dict] = []
    hd = ["metric"]
    for fld in folders:
        if quick:
            stats = Stats()
            stats.load_json(read_json(fld + "/stats.txt"))
            entries = read_json(fld + "/entries.txt")
        else:
            stats, entries = main(fld)
        entries["name"] = stats.meta["name"]
        ents.append(entries)
    ents.sort(key=lambda e: e["TBM avg (us)"])
    hd += [e["name"] for e in ents]
    rows = [hd]
    for k in ents[0].keys():
        if k == "name":
            continue
        line = [k]
        for e in ents:
            line.append(f"{e[k]:,}")
        rows.append(line)
    tbl = print_markdown_table(rows)

    with open("README.md", "r") as f:
        rdm = f.read()
        rdm = re.sub(
            r"<!-- GEN-BEGIN -->.*?<!-- GEN-END -->",
            "<!-- GEN-BEGIN -->\n" + tbl + "<!-- GEN-END -->",
            rdm,
            flags=re.DOTALL,
        )
    with open("README.md", "w") as f:
        f.write(rdm)

    plot_metrics(
        ents,
        prefix="TBM ",
        title="Per-token mask computation time (Time Between Masks aka TBM)",
    )
    plot_metrics(
        ents,
        prefix="TTFM ",
        title="Grammar compilation (Time To First Mask aka TTFM); 900s timeout",
    )
