#!/usr/bin/env python3
"""Precompute transformer-ready pipeline graphs."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch_geometric.data import HeteroData

from halide_gnn_cost_model.ast_parser import build_ast_node_type_vocab
from halide_gnn_cost_model.data import (
    PREPROCESSED_FILENAME,
    TRANSFORMER_PREPROCESSED_FILENAME,
    PipelineDataset,
    graph_transformer_preprocessor,
    load_pipeline,
)
from halide_gnn_cost_model.schedule_parser import build_schedule_node_type_vocab


logger = logging.getLogger(__name__)


def _load_base_graph(
    pipeline_dir: Path,
    ast_vocab,
    sched_vocab,
) -> HeteroData:
    preprocessed_path = pipeline_dir / PREPROCESSED_FILENAME
    if preprocessed_path.exists():
        return PipelineDataset._load_preprocessed(preprocessed_path)
    return load_pipeline(pipeline_dir, ast_vocab=ast_vocab, sched_vocab=sched_vocab)


def preprocess_pipeline(
    pipeline_dir: Path,
    *,
    ast_vocab,
    sched_vocab,
    k: int,
    walk_length: int,
    max_degree: int,
    force: bool = False,
) -> Path:
    """Write the transformer-friendly representation for a single pipeline."""

    if not pipeline_dir.exists() or not pipeline_dir.is_dir():
        raise ValueError(
            f"Pipeline directory {pipeline_dir} does not exist or is not a directory."
        )

    output_path = pipeline_dir / TRANSFORMER_PREPROCESSED_FILENAME
    if output_path.exists() and not force:
        logger.debug(
            "Skipping %s; transformer graph already exists.", pipeline_dir.name
        )
        return output_path

    hetero_graph = _load_base_graph(pipeline_dir, ast_vocab, sched_vocab)
    transformer_graph = graph_transformer_preprocessor(
        hetero_graph, k=k, walk_length=walk_length, max_degree=max_degree
    )
    torch.save(transformer_graph, output_path)
    logger.info("Saved transformer graph to %s", output_path)
    return output_path


def preprocess_pipelines(
    pipelines_dir: Path,
    *,
    k: int,
    walk_length: int,
    max_degree: int,
    force: bool = False,
) -> Tuple[int, int]:
    """Preprocess every pipeline directory under ``pipelines_dir``."""

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
        output_path = pipeline_dir / TRANSFORMER_PREPROCESSED_FILENAME
        if output_path.exists() and not force:
            logger.debug(
                "Skipping %s; transformer graph already exists.", pipeline_dir.name
            )
            skipped += 1
            continue

        preprocess_pipeline(
            pipeline_dir,
            ast_vocab=ast_vocab,
            sched_vocab=sched_vocab,
            k=k,
            walk_length=walk_length,
            max_degree=max_degree,
            force=force,
        )
        processed += 1

    return processed, skipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess pipelines into transformer-ready graphs."
    )
    parser.add_argument(
        "--pipelines-dir",
        type=Path,
        default=Path("pipelines"),
        help="Directory containing pipeline subdirectories (default: ./pipelines).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of Laplacian eigenvectors to retain (default: 8).",
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=8,
        help="Random walk positional encoding length (default: 8).",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=7,
        help="Maximum degree for one-hot encoding (default: 7).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate files even if they already exist.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        processed, skipped = preprocess_pipelines(
            args.pipelines_dir,
            k=args.k,
            walk_length=args.walk_length,
            max_degree=args.max_degree,
            force=args.force,
        )
    except Exception:  # pragma: no cover - surface context at CLI level
        logger.exception("Failed to preprocess pipelines for transformer mode")
        sys.exit(1)

    logger.info(
        "Transformer preprocessing complete: %d processed, %d skipped (force=%s)",
        processed,
        skipped,
        args.force,
    )


if __name__ == "__main__":
    main()
