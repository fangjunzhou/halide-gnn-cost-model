#!/usr/bin/env python3
"""
scripts/run_pipebench.py

Script to run PipeBench benchmarks and collect performance data.

Usage:
  python3 scripts/run_pipebench.py
"""

import os
import subprocess
from pathlib import Path
import argparse
import re
import logging


def run_pipebench(benchmark_path: Path, output_path: Path) -> int:
    """Run a single PipeBench benchmark and collect performance data.

    Returns the subprocess return code.
    """
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark executable not found at {benchmark_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Construct command: pass benchmark_out as a single arg
    cmd = [str(benchmark_path), f"--benchmark_out={str(output_path)}"]
    logging.info("Running: %s", " ".join(cmd))

    # Capture and stream output
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    # Print benchmark output to stdout so user sees progress
    print(proc.stdout, end="")
    return proc.returncode


def discover_and_run_all(
    bin_dir: Path, pipelines_dir: Path, pattern: str = "pipebench_pipeline_"
) -> int:
    """Discover all executables in bin_dir matching pattern and run them.

    Mapping rule: an executable named `pipebench_pipeline_X` writes output to
    `pipelines/pipeline_X/benchmark.json`.

    Returns 0 if all ran successfully, otherwise returns number of failures (>0).
    """
    if not bin_dir.exists():
        raise FileNotFoundError(f"Binary directory not found: {bin_dir}")

    exes = sorted(bin_dir.glob(f"{pattern}*"))
    # Filter to files that are executable (and skip directories like 'pipegen')
    exes = [p for p in exes if p.is_file() and os.access(p, os.X_OK)]

    if not exes:
        logging.warning("No executables matching '%s' found in %s", pattern, bin_dir)
        return 0

    failures = 0
    name_re = re.compile(r"^pipebench_(pipeline_\d+)$")

    for exe in exes:
        m = name_re.match(exe.name)
        if not m:
            logging.info("Skipping unexpected file: %s", exe.name)
            continue

        pipeline_name = m.group(1)  # e.g. pipeline_0
        out_path = pipelines_dir / pipeline_name / "benchmark.json"

        try:
            rc = run_pipebench(exe, out_path)
            if rc != 0:
                logging.error("Benchmark %s exited with code %d", exe.name, rc)
                failures += 1
        except Exception as e:
            logging.exception("Failed to run %s: %s", exe, e)
            failures += 1

    return failures


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run all PipeBench pipeline benchmarks and save outputs"
    )
    p.add_argument(
        "--bin-dir",
        type=Path,
        default=Path("build/bin"),
        help="Directory containing pipebench executables",
    )
    p.add_argument(
        "--pipelines-dir",
        type=Path,
        default=Path("pipelines"),
        help="Pipelines directory to write benchmark.json into",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="pipebench_pipeline_",
        help="Filename prefix for benchmark executables",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    failures = discover_and_run_all(args.bin_dir, args.pipelines_dir, args.pattern)
    if failures:
        logging.error("Completed with %d failures", failures)
        raise SystemExit(1)
    logging.info("All benchmarks completed successfully")


if __name__ == "__main__":
    main()
