#!/usr/bin/env python3
"""
Script to build and run pipeline generation.
"""

import subprocess
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def build_pipegen():
    """Build the pipegen executable using cmake."""
    logger.info("Building pipegen executable...")
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TARGET=pipegen"],
        cwd=build_dir,
        check=True,
    )
    subprocess.run(["make"], cwd=build_dir, check=True)
    logger.info("Build completed.")


def run_pipegen(dag_seed: int, schedule_seed: int, pipeline_dir: Path):
    """Run the pipegen executable with specified parameters.

    :param dag_seed: Seed for DAG generation.
    :param schedule_seed: Seed for schedule generation.
    :param output_dir: Directory to save generated pipelines.
    """
    logger.info(
        f"Running pipegen with dag_seed={dag_seed}, schedule_seed={schedule_seed}, output_dir={pipeline_dir}"
    )
    build_dir = Path("build")
    pipegen_executable = build_dir / "bin" / "pipegen"
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


def generate_pipelines(dag_seed: int, num_schedules: int, output_dir: Path):
    """Generate a set of pipelines by building pipegen then invoking it.

    :param dag_seed: Seed for DAG generation.
    :param num_schedules: Number of schedules to generate.
    :param output_dir: Directory to save generated pipelines.
    """
    # Build the pipegen executable
    build_pipegen()

    # Generate pipelines
    for schedIdx in range(num_schedules):
        pipeline_dir = output_dir / f"pipeline_{dag_seed}_{schedIdx}"
        run_pipegen(dag_seed, schedIdx, pipeline_dir)
