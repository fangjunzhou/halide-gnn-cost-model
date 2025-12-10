#!/usr/bin/env python3
"""Preprocess PipeBench pipelines into serialized HeteroData graphs."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import torch

from halide_gnn_cost_model.ast_parser import build_ast_node_type_vocab
from halide_gnn_cost_model.data import PREPROCESSED_FILENAME, load_pipeline
from halide_gnn_cost_model.schedule_parser import build_schedule_node_type_vocab


logger = logging.getLogger(__name__)


def preprocess_pipeline(
    pipeline_dir: Path,
    ast_vocab=None,
    sched_vocab=None,
    force: bool = False,
) -> Path:
    """Serialize the pipeline graph for quick loading during training."""

    if not pipeline_dir.exists() or not pipeline_dir.is_dir():
        raise ValueError(
            f"Pipeline directory {pipeline_dir} does not exist or is not a directory."
        )

    output_path = pipeline_dir / PREPROCESSED_FILENAME
    if output_path.exists() and not force:
        logger.debug(
            "Skipping %s; preprocessed graph already exists.", pipeline_dir.name
        )
        return output_path

    data = load_pipeline(pipeline_dir, ast_vocab, sched_vocab)
    torch.save(data, output_path)
    logger.info("Saved preprocessed graph to %s", output_path)
    return output_path


def preprocess_pipelines(pipelines_dir: Path, force: bool = False) -> Tuple[int, int]:
    """Preprocess every pipeline directory under ``pipelines_dir``.

    Returns a tuple of (processed_count, skipped_count).
    """

    if not pipelines_dir.exists() or not pipelines_dir.is_dir():
        raise ValueError(
            f"Pipelines directory {pipelines_dir} does not exist or is not a directory."
        )

    pipeline_dirs = sorted(
        [d for d in pipelines_dir.iterdir() if d.is_dir()], key=lambda p: p.name
    )

    if not pipeline_dirs:
        logger.info("No pipeline directories found in %s", pipelines_dir)
        return 0, 0

    ast_vocab = build_ast_node_type_vocab()
    sched_vocab = build_schedule_node_type_vocab()

    processed = 0
    skipped = 0

    for pipeline_dir in pipeline_dirs:
        output_path = pipeline_dir / PREPROCESSED_FILENAME
        if output_path.exists() and not force:
            logger.debug(
                "Skipping %s; preprocessed graph already exists.", pipeline_dir.name
            )
            skipped += 1
            continue

        preprocess_pipeline(
            pipeline_dir,
            ast_vocab=ast_vocab,
            sched_vocab=sched_vocab,
            force=force,
        )
        processed += 1

    return processed, skipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess PipeBench pipelines into serialized graph data."
    )
    parser.add_argument(
        "--pipelines-dir",
        type=Path,
        default=Path("pipelines"),
        help="Directory containing pipeline subdirectories (default: ./pipelines).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate preprocessed files even if they already exist.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging for troubleshooting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        processed, skipped = preprocess_pipelines(args.pipelines_dir, force=args.force)
    except Exception:  # pragma: no cover - propagate meaningful context to CLI
        logger.exception("Failed to preprocess pipelines")
        sys.exit(1)

    logger.info(
        "Preprocessing complete: %d processed, %d skipped (force=%s)",
        processed,
        skipped,
        args.force,
    )


if __name__ == "__main__":
    main()
