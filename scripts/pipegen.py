#!/usr/bin/env python3
"""
Script to build and run pipeline generation.
"""

import os
import subprocess
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def build_pipegen(build_dir: Path | None = None):
    """Build the pipegen executable using cmake.

    :param build_dir: Directory to use for out-of-source build. If None,
                      defaults to ./build relative to the repository root.
    """
    logger.info("Building pipegen executable...")
    if build_dir is None:
        build_dir = Path("build")
    build_dir = build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    # Resolve repository root based on the script location (one level up from
    # the scripts/ directory).
    source_dir = Path(__file__).resolve().parents[1]

    # Configure the build using modern CMake -S/-B flags so absolute build
    # directories outside the repo (e.g. /var/tmp/pipe-build) work correctly.
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TARGET=pipegen",
        ],
        check=True,
    )

    # Build the target using cmake --build for portability across generators.
    # Use the --parallel option to leverage multiple CPU cores if available.
    parallel = os.cpu_count() or 1
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            str(parallel),
        ],
        check=True,
    )
    logger.info("Build completed.")


def run_pipegen(
    dag_seed: int, schedule_seed: int, pipeline_dir: Path, build_dir: Path | None = None
):
    """Run the pipegen executable with specified parameters.

    :param dag_seed: Seed for DAG generation.
    :param schedule_seed: Seed for schedule generation.
    :param output_dir: Directory to save generated pipelines.
    """
    logger.info(
        f"Running pipegen with dag_seed={dag_seed}, schedule_seed={schedule_seed}, output_dir={pipeline_dir}"
    )
    if build_dir is None:
        build_dir = Path("build")
    pipegen_executable = Path(build_dir) / "bin" / "pipegen"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(pipegen_executable),
            "--dag-seed",
            str(dag_seed),
            "--schedule-seed",
            str(schedule_seed),
            "--output-dir",
            str(pipeline_dir),
        ],
        check=True,
    )
    logger.info("Pipegen run completed.")


def generate_pipelines(
    dag_seed: int, num_schedules: int, output_dir: Path, build_dir: Path | None = None
):
    """Generate a set of pipelines by building pipegen then invoking it.

    :param dag_seed: Seed for DAG generation.
    :param num_schedules: Number of schedules to generate.
    :param output_dir: Directory to save generated pipelines.
    """
    # Build the pipegen executable
    build_pipegen(build_dir)

    # Generate pipelines
    for schedIdx in range(num_schedules):
        pipeline_dir = output_dir / f"pipeline_{dag_seed}_{schedIdx}"
        run_pipegen(dag_seed, schedIdx, pipeline_dir, build_dir)
