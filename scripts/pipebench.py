#!/usr/bin/env python3
"""
Script to run PipeBench benchmarks and collect performance data.
"""

import os
import subprocess
from pathlib import Path
import re
import logging


logger = logging.getLogger(__name__)


def build_pipebench(pipelines_dir: Path | None = None):
    """Build the pipebench executables using cmake.

    If `pipelines_dir` is provided, pass it to CMake as the PIPES_DIR cache
    variable so the CMakeLists can pick up pipeline dirs from that location.
    """
    logger.info("Building pipebench executables...")
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)

    cmake_cmd = [
        "cmake",
        "..",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TARGET=pipebench",
    ]
    if pipelines_dir is not None:
        cmake_cmd.append(f"-DPIPES_DIR={str(pipelines_dir.resolve())}")

    subprocess.run(cmake_cmd, cwd=build_dir, check=True)
    subprocess.run(["make"], cwd=build_dir, check=True)
    logger.info("Build completed.")


def run_pipebench(benchmark_path: Path, output_path: Path) -> int:
    """Run a single PipeBench benchmark and collect performance data.

    Returns the subprocess return code.
    """
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark executable not found at {benchmark_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Capture and stream output
    proc = subprocess.run(
        [str(benchmark_path), f"--benchmark_out={str(output_path)}"], check=True
    )
    return proc.returncode


def benchmark_pipelines(pipelines_dir: Path) -> int:
    """Benchmark all pipelines found in `pipelines_dir`.

    This function builds the pipebench binaries (in `build/bin`) and then for
    each pipeline directory named `pipeline_<dag>_<sched>` it looks for an
    executable named `pipebench_pipeline_<dag>_<sched>` in `build/bin` and
    runs it, writing output to `pipelines/pipeline_<dag>_<sched>/benchmark.json`.

    Returns 0 if all ran successfully, otherwise returns number of failures (>0).
    """
    bin_dir = Path("build") / "bin"
    if not bin_dir.exists():
        raise FileNotFoundError(f"Binary directory not found: {bin_dir}")

    # Build the pipebench executables
    build_pipebench(pipelines_dir)

    failures = 0

    if not pipelines_dir.exists():
        logger.warning("Pipelines directory does not exist: %s", pipelines_dir)
        return 0

    for entry in sorted(pipelines_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("pipeline_"):
            continue

        exe_name = f"pipebench_{entry.name}"
        exe_path = bin_dir / exe_name
        out_path = entry / "benchmark.json"

        if not exe_path.exists() or not os.access(exe_path, os.X_OK):
            logger.error("Executable not found or not executable: %s", exe_path)
            failures += 1
            continue

        try:
            logger.info("Running benchmark: %s", exe_path.name)
            rc = run_pipebench(exe_path, out_path)
            if rc != 0:
                logger.error("Benchmark %s exited with code %d", exe_path.name, rc)
                failures += 1
        except Exception as e:
            logger.exception("Failed to run %s: %s", exe_path, e)
            failures += 1

    return failures
