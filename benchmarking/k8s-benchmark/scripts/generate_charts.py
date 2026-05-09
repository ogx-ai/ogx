#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# /// script
# dependencies = [
#   "matplotlib",
# ]
# ///
"""
Script to generate benchmark charts from guidellm text results.
Creates 2x2 grid charts with RPS, Request Latency, TTFT, and ITL metrics against concurrent@x values.
Outputs one chart file per vLLM replica group, with each line representing one benchmark run.
"""

import glob
import os
import re

import matplotlib.pyplot as plt


def extract_setup_name(filename: str) -> str:
    """Extract setup name from filename and format legend appropriately."""
    basename = os.path.basename(filename)

    # Try new pattern: guidellm-benchmark-stack-s{stack_replicas}-sw{workers}-v{vllm_replicas}-{timestamp}.txt
    match = re.search(r"guidellm-benchmark-stack-s(\d+)-sw(\d+)-v(\d+)-(\d{8})-(\d{6})\.txt", basename)
    if match:
        stack_replicas = match.group(1)
        workers = match.group(2)
        vllm_replicas = match.group(3)
        date = match.group(4)
        time = match.group(5)
        return f"stack-s{stack_replicas}-sw{workers}-v{vllm_replicas}"

    # Try new vLLM pattern: guidellm-benchmark-vllm-v{vllm_replicas}-{timestamp}.txt
    match = re.search(r"guidellm-benchmark-vllm-v(\d+)-(\d{8})-(\d{6})\.txt", basename)
    if match:
        vllm_replicas = match.group(1)
        date = match.group(2)
        time = match.group(3)
        return f"vllm-v{vllm_replicas}"

    # Fall back to old pattern: guidellm-benchmark-{target}-{stack_replicas}-w{workers}-{vllm_replicas}-{timestamp}.txt
    match = re.search(r"guidellm-benchmark-([^-]+)-(\d+)-w(\d+)-(\d+)-(\d+)-(\d+)\.txt", basename)
    if match:
        target = match.group(1)
        stack_replicas = match.group(2)
        workers = match.group(3)
        vllm_replicas = match.group(4)
        date = match.group(5)
        time = match.group(6)

        if target == "vllm":
            return f"vllm-{vllm_replicas}-w{workers}-{vllm_replicas}"
        else:
            return f"stack-replicas{stack_replicas}-w{workers}-vllm-replicas{vllm_replicas}-{date}-{time}"

    # Fall back to older pattern: guidellm-benchmark-{target}-{stack_replicas}-{vllm_replicas}-{timestamp}.txt
    match = re.search(r"guidellm-benchmark-([^-]+)-(\d+)-(\d+)-(\d+)-(\d+)\.txt", basename)
    if match:
        target = match.group(1)
        stack_replicas = match.group(2)
        vllm_replicas = match.group(3)
        date = match.group(4)
        time = match.group(5)

        if target == "vllm":
            return f"vllm-{vllm_replicas}-w1-{vllm_replicas}"
        else:
            return f"stack-replicas{stack_replicas}-vllm-replicas{vllm_replicas}-{date}-{time}"

    return basename.replace("guidellm-benchmark-", "").replace(".txt", "")


def parse_txt_file(filepath: str) -> list[tuple[float, float, float, float, float, str]]:
    """
    Parse a text benchmark file and extract concurrent@x, RPS, TTFT, ITL, and request latency data.
    Returns list of (concurrency, rps_mean, ttft_mean, itl_mean, req_latency_mean, setup_name) tuples.
    """
    setup_name = extract_setup_name(filepath)
    data_points = []

    try:
        with open(filepath) as f:
            content = f.read()

        # Find the benchmark stats table
        lines = content.split("\n")
        in_stats_table = False
        header_lines_seen = 0

        for line in lines:
            line_stripped = line.strip()

            # Look for the start of the stats table
            if "Benchmarks Stats:" in line:
                in_stats_table = True
                continue

            if in_stats_table:
                # Skip the first few separator/header lines
                if line_stripped.startswith("=") or line_stripped.startswith("-"):
                    header_lines_seen += 1
                    if header_lines_seen >= 3:  # After seeing multiple header lines, look for concurrent@ data
                        if line_stripped.startswith("=") and "concurrent@" not in line_stripped:
                            break
                    continue

            # Parse concurrent@ lines in the stats table (may have leading spaces)
            if in_stats_table and "concurrent@" in line:
                parts = [part.strip() for part in line.split("|")]

                if len(parts) >= 12:  # Make sure we have enough columns for new format
                    try:
                        # Extract concurrency from benchmark name (e.g., concurrent@1 -> 1)
                        concurrent_match = re.search(r"concurrent@(\d+)", parts[0])
                        if not concurrent_match:
                            continue
                        concurrency = float(concurrent_match.group(1))

                        # Extract metrics from the new table format
                        # From your image, the table has these columns with | separators:
                        # Benchmark | Per Second | Concurrency | Out Tok/sec | Tot Tok/sec | Req Latency (sec) | TTFT (ms) | ITL (ms) | TPOT (ms)
                        # Looking at the mean/median/p99 structure, need to find the mean columns
                        # The structure shows: mean | median | p99 for each metric
                        rps_mean = float(parts[1])  # Per Second (RPS)
                        req_latency_mean = float(parts[6]) * 1000  # Request latency mean (convert from sec to ms)
                        ttft_mean = float(parts[9])  # TTFT mean column
                        itl_mean = float(parts[12])  # ITL mean column

                        data_points.append((concurrency, rps_mean, ttft_mean, itl_mean, req_latency_mean, setup_name))

                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line '{line}' in {filepath}: {e}")
                        continue

    except (OSError, FileNotFoundError) as e:
        print(f"Error reading {filepath}: {e}")

    return data_points


def generate_charts(benchmark_dir: str = "results"):
    """Generate 2x2 grid charts (RPS, Request Latency, TTFT, ITL) from benchmark text files."""
    # Find all text result files instead of JSON
    txt_pattern = os.path.join(benchmark_dir, "guidellm-benchmark-*.txt")
    txt_files = glob.glob(txt_pattern)

    if not txt_files:
        print(f"No text files found matching pattern: {txt_pattern}")
        return

    print(f"Found {len(txt_files)} text files")

    # Parse all files and collect data
    all_data = {}  # setup_name -> [(concurrency, rps, ttft, itl, req_latency), ...]

    for txt_file in txt_files:
        print(f"Processing {txt_file}")
        data_points = parse_txt_file(txt_file)

        for concurrency, rps, ttft, itl, req_latency, setup_name in data_points:
            if setup_name not in all_data:
                all_data[setup_name] = []
            all_data[setup_name].append((concurrency, rps, ttft, itl, req_latency))

    if not all_data:
        print("No data found to plot")
        return

    # Sort data points by concurrency for each setup
    for setup_name in all_data:
        all_data[setup_name].sort(key=lambda x: x[0])  # Sort by concurrency

    # Group setups by vLLM replica number (original approach)
    replica_groups = {}  # vllm_replica_count -> {setup_name: points}

    for setup_name, points in all_data.items():
        # Extract vLLM replica number from setup name
        # Expected formats:
        # - New stack format: "stack-s{X}-sw{W}-v{Y}"
        # - New vLLM format: "vllm-v{Y}"
        # - Old formats: "stack-replicas{X}-w{W}-vllm-replicas{Y}" or "vllm-{Y}-w{W}-{Y}"

        # Try new formats first
        vllm_match = re.search(r"-v(\d+)$", setup_name)  # Matches both "stack-s1-sw2-v3" and "vllm-v1"
        if not vllm_match:
            # Try old stack format
            vllm_match = re.search(r"vllm-replicas(\d+)", setup_name)
        if not vllm_match:
            # Try old vLLM format: "vllm-{Y}-w{W}-{Y}"
            vllm_match = re.search(r"vllm-(\d+)-w\d+-\d+", setup_name)

        if vllm_match:
            vllm_replica_num = int(vllm_match.group(1))
            if vllm_replica_num not in replica_groups:
                replica_groups[vllm_replica_num] = {}
            replica_groups[vllm_replica_num][setup_name] = points
        else:
            print(f"Warning: Could not extract vLLM replica count from setup name: {setup_name}")

    def create_charts(data_dict, prefix, title_prefix):
        """Create a 2x2 grid with RPS, Request Latency, TTFT, and ITL charts."""
        if not data_dict:
            print(f"No data found for {prefix}")
            return

        # Create 2x2 subplot grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{title_prefix} Benchmark Results", fontsize=16, fontweight="bold")

        # Collect all unique concurrency values for tick setting
        all_concurrency_values = set()
        for points in data_dict.values():
            all_concurrency_values.update([p[0] for p in points])
        all_concurrency_values = sorted(all_concurrency_values)

        # Plot data for each setup in alphabetical order
        for setup_name in sorted(data_dict.keys()):
            points = data_dict[setup_name]
            if not points:
                continue

            concurrency_values = [p[0] for p in points]
            rps_values = [p[1] for p in points]
            ttft_values = [p[2] for p in points]
            itl_values = [p[3] for p in points]
            req_latency_values = [p[4] for p in points]

            # RPS chart (top-left)
            ax1.plot(concurrency_values, rps_values, marker="o", label=setup_name, linewidth=2, markersize=6)

            # Request Latency chart (top-right)
            ax2.plot(concurrency_values, req_latency_values, marker="o", label=setup_name, linewidth=2, markersize=6)

            # TTFT chart (bottom-left)
            ax3.plot(concurrency_values, ttft_values, marker="o", label=setup_name, linewidth=2, markersize=6)

            # ITL chart (bottom-right)
            ax4.plot(concurrency_values, itl_values, marker="o", label=setup_name, linewidth=2, markersize=6)

        # Configure all charts after plotting data
        axes = [ax1, ax2, ax3, ax4]
        titles = ["RPS", "Request Latency", "TTFT", "ITL"]
        ylabels = [
            "Requests Per Second (RPS)",
            "Request Latency (ms)",
            "Time to First Token (ms)",
            "Inter Token Latency (ms)",
        ]

        for ax, title, ylabel in zip(axes, titles, ylabels, strict=False):
            ax.set_xlabel("Concurrency", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xscale("log", base=2)
            ax.set_xticks(all_concurrency_values)
            ax.set_xticklabels([str(int(x)) for x in all_concurrency_values])
            ax.grid(True, alpha=0.3)

        # Add legend to the right-most subplot (top-right)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        # Save the combined chart
        combined_filename = os.path.join(benchmark_dir, f"{prefix}_benchmark_results.png")
        plt.savefig(combined_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Combined benchmark chart saved to {combined_filename}")

    # Print grouping information
    for replica_count, data_dict in replica_groups.items():
        print(f"vLLM Replica {replica_count} setups: {list(data_dict.keys())}")

    # Create separate charts for each replica group
    for replica_count, data_dict in replica_groups.items():
        prefix = f"vllm_replica{replica_count}"
        title = f"vLLM Replicas={replica_count}"
        create_charts(data_dict, prefix, title)

    # Print summary
    print("\nSummary:")
    for setup_name, points in all_data.items():
        print(f"{setup_name}: {len(points)} data points")


if __name__ == "__main__":
    generate_charts()
