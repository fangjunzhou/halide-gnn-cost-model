#!/usr/bin/env python3
"""
Singularity container entry point for building and running all PipeBench pipeline benchmarks.
"""

import logging
from pathlib import Path
import argparse
import sys

import pipegen
import pipebench
import pipepreprocess


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate pipelines with pipegen and run PipeBench benchmarks."
    )
    parser.add_argument(
        "--pipeline-id",
        type=int,
        default=0,
        help="Pipeline ID (used as dag seed).",
    )
    parser.add_argument(
        "--num-schedules",
        type=int,
        default=16,
        help="Number of schedules to generate per pipeline.",
    )
    parser.add_argument(
        "--pipelines-dir",
        type=Path,
        default=Path("pipelines"),
        help="Directory containing pipeline definitions.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Out-of-source build directory to use for CMake (default: ./build).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    pipeline_id: int = args.pipeline_id
    num_schedules: int = args.num_schedules
    pipelines_dir: Path = args.pipelines_dir
    build_dir: Path = args.build_dir

    logger.info(
        "Generating pipeline %s with %s schedules into %s",
        pipeline_id,
        num_schedules,
        pipelines_dir,
    )
    try:
        pipegen.generate_pipelines(pipeline_id, num_schedules, pipelines_dir, build_dir)
    except Exception:
        logger.exception("Failed to generate pipelines")
        sys.exit(1)

    logger.info("Benchmarking generated pipelines in %s", pipelines_dir)
    try:
        failures = pipebench.benchmark_pipelines(pipelines_dir, build_dir)
    except Exception:
        logger.exception("Failed to run benchmarks")
        sys.exit(1)

    logger.info("Preprocessing pipelines in %s", pipelines_dir)
    try:
        processed, skipped = pipepreprocess.preprocess_pipelines(
            pipelines_dir, force=True
        )
    except Exception:
        logger.exception("Failed to preprocess pipelines")
        sys.exit(1)

    logger.info(
        "Pipeline preprocessing complete (processed=%d, skipped=%d)",
        processed,
        skipped,
    )

    if failures:
        logger.error("Completed with %d failures", failures)
        sys.exit(1)

    logger.info("All pipelines generated and benchmarked successfully")


if __name__ == "__main__":
    main()
