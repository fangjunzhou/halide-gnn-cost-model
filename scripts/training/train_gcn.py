"""Train a PipelineModel with a GraphSAGE encoder on Halide pipeline graphs.

This script mirrors the workflow from ``notebooks/gnn_playground.ipynb`` while
adding a configurable command-line interface so hyperparameters and paths can
be adjusted without editing code.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_hetero

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None  # type: ignore

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter as SummaryWriterType
else:  # pragma: no cover - runtime fallback when tensorboard missing
    SummaryWriterType = Any

from tqdm.auto import tqdm

from halide_gnn_cost_model.data import PipelineDataset
from halide_gnn_cost_model.model import PipeGCN, PipelineModel


LOGGER = logging.getLogger(__name__)
DEFAULT_PIPELINES_DIR = Path("resources/pipelines-16k-data")
DEFAULT_MODELS_DIR = Path("resources/models/gcn")
DEFAULT_LOG_DIR = Path("resources/runs")
TENSORBOARD_UNAVAILABLE_MSG = (
    "TensorBoard is unavailable; install tensorboard or run without --log-dir."
)
EPS = 1e-8


@dataclass
class TrainMetrics:
    loss: float
    runtime_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GCN-based pipeline runtime regressor."
    )
    parser.add_argument(
        "--pipelines-dir",
        type=Path,
        default=DEFAULT_PIPELINES_DIR,
        help="Directory containing preprocessed pipeline graphs.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory where model checkpoints will be saved.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Base directory for TensorBoard logs (ignored if TensorBoard missing).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name used for the log directory. Defaults to timestamp.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Computation device to use.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Evaluation batch size (defaults to --batch-size).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Hidden size within the GCN encoder.",
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        default=64,
        help="Output size of the GCN encoder (also the PipelineModel input size).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of layers in the GCN encoder.",
    )
    parser.add_argument(
        "--gnn-dropout",
        type=float,
        default=0.5,
        help="Dropout rate applied inside the GCN encoder.",
    )
    parser.add_argument(
        "--model-hidden",
        type=int,
        default=32,
        help="Hidden size within the PipelineModel MLP head.",
    )
    parser.add_argument(
        "--model-dropout",
        type=float,
        default=0.1,
        help="Dropout applied in the PipelineModel MLP head.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for Adam optimizer.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Checkpoint interval in epochs.",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Always write best validation checkpoint to 'best.pt'.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.95,
        help="Fraction of data used for training (rest used for evaluation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Clip gradient norm to this value (disabled if None).",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional path to save run metadata as JSON.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available.")

    if torch.cuda.is_available():
        LOGGER.info("Using CUDA device")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        LOGGER.info("Using MPS device")
        return torch.device("mps")
    LOGGER.info("Falling back to CPU device")
    return torch.device("cpu")


def prepare_datasets(
    dataset: PipelineDataset,
    train_split: float,
    seed: int,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("--train-split must be between 0 and 1 (exclusive).")

    total = len(dataset)
    if total < 2:
        raise ValueError("Dataset must contain at least two samples for a split.")

    train_size = max(1, int(total * train_split))
    eval_size = total - train_size
    if eval_size == 0:
        train_size -= 1
        eval_size = 1

    generator = torch.Generator().manual_seed(seed)
    train_subset, eval_subset = random_split(
        dataset, [train_size, eval_size], generator=generator
    )
    LOGGER.info("Dataset split: %s train / %s eval", train_size, eval_size)
    return train_subset, eval_subset


def make_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
    batch_size: int,
    eval_batch_size: Optional[int],
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size or batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, eval_loader


def create_model(
    data_example: torch.Tensor,
    dataset: PipelineDataset,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    gnn_dropout: float,
    model_hidden: int,
    model_dropout: float,
) -> PipelineModel:
    encoder = PipeGCN(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=gnn_dropout,
    )
    encoder = to_hetero(encoder, data_example.metadata(), aggr="sum")

    num_runtime_targets = int(data_example.y.numel())
    model = PipelineModel(
        encoder,
        feat_channels=out_channels,
        num_runtime_targets=num_runtime_targets,
        ast_vocab_size=len(dataset.ast_vocab),
        sched_vocab_size=len(dataset.sched_vocab),
        hidden_channels=model_hidden,
        dropout=model_dropout,
    )
    return model


def compute_metrics(
    model: PipelineModel,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> TrainMetrics:
    model.eval()
    loss_total = 0.0
    batch_count = 0
    rel_err_sum = 0.0
    rel_err_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ptr = getattr(batch["loop_level"], "ptr", None)
            preds = model(batch, ptr)
            targets = torch.log(batch.y.clamp_min(EPS))
            loss_total += loss_fn(preds.reshape(-1), targets).item()
            batch_count += 1

            preds_runtime = torch.exp(preds)
            rel_error = torch.abs(
                preds_runtime.reshape(-1) - batch.y
            ) / batch.y.clamp_min(EPS)
            rel_err_sum += rel_error.sum().item()
            rel_err_count += rel_error.numel()

    if batch_count == 0:
        return TrainMetrics(loss=0.0, runtime_error=0.0)

    avg_loss = loss_total / batch_count
    avg_rel_error = rel_err_sum / max(rel_err_count, 1)
    return TrainMetrics(loss=avg_loss, runtime_error=avg_rel_error)


def save_checkpoint(
    models_dir: Path,
    model: PipelineModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: TrainMetrics,
    eval_metrics: TrainMetrics,
    args: argparse.Namespace,
    best: bool = False,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    suffix = "best" if best else f"epoch_{epoch}"
    checkpoint_path = models_dir / f"pipeline_model_{suffix}.pt"
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": asdict(train_metrics),
        "eval_metrics": asdict(eval_metrics),
        "args": vars(args),
    }
    torch.save(payload, checkpoint_path)
    LOGGER.info("Saved checkpoint to %s", checkpoint_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    set_deterministic(args.seed)

    pipelines_dir = args.pipelines_dir.expanduser()
    models_dir = args.models_dir.expanduser()
    log_dir = args.log_dir.expanduser() if args.log_dir else None

    dataset = PipelineDataset(pipelines_dir)
    example_data = dataset[0]

    train_dataset, eval_dataset = prepare_datasets(dataset, args.train_split, args.seed)
    train_loader, eval_loader = make_dataloaders(
        train_dataset,
        eval_dataset,
        args.batch_size,
        args.eval_batch_size,
        args.num_workers,
    )

    device = select_device(args.device)
    LOGGER.info("Training on device: %s", device)

    model = create_model(
        example_data,
        dataset,
        args.hidden_channels,
        args.out_channels,
        args.num_layers,
        args.gnn_dropout,
        args.model_hidden,
        args.model_dropout,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()

    writer: Optional[SummaryWriterType] = None
    if log_dir is not None and SummaryWriter is not None:
        run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
        full_log_dir = log_dir / run_name
        full_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(full_log_dir))
        LOGGER.info("Logging TensorBoard metrics to %s", full_log_dir)
    elif log_dir is not None and SummaryWriter is None:
        LOGGER.warning(TENSORBOARD_UNAVAILABLE_MSG)

    best_eval_loss = float("inf")
    best_eval_metrics: Optional[TrainMetrics] = None
    global_step = 0

    progress_disable = args.no_progress_bar

    try:
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            loader_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{args.num_epochs}",
                leave=False,
                disable=progress_disable,
            )
            for batch in loader_iter:
                batch = batch.to(device)
                ptr = getattr(batch["loop_level"], "ptr", None)
                optimizer.zero_grad()
                preds = model(batch, ptr)
                targets = torch.log(batch.y.clamp_min(EPS))
                loss = loss_fn(preds.reshape(-1), targets)
                loss.backward()
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                if writer is not None:
                    writer.add_scalar("Loss/train_step", loss.item(), global_step)
                global_step += 1

            train_metrics = compute_metrics(model, train_loader, device, loss_fn)
            eval_metrics = compute_metrics(model, eval_loader, device, loss_fn)

            LOGGER.info(
                "Epoch %s | train loss %.4f | eval loss %.4f | eval rel err %.2f%%",
                epoch,
                train_metrics.loss,
                eval_metrics.loss,
                eval_metrics.runtime_error * 100.0,
            )

            if writer is not None:
                writer.add_scalar("Loss/train_epoch", train_metrics.loss, epoch)
                writer.add_scalar("Loss/eval_epoch", eval_metrics.loss, epoch)
                writer.add_scalar(
                    "Eval/avg_relative_error_pct",
                    eval_metrics.runtime_error * 100.0,
                    epoch,
                )

            if args.metadata_json:
                metadata = {
                    "epoch": epoch,
                    "train_metrics": asdict(train_metrics),
                    "eval_metrics": asdict(eval_metrics),
                }
                args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
                with open(args.metadata_json, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

            if epoch % args.save_every == 0:
                save_checkpoint(
                    models_dir,
                    model,
                    optimizer,
                    epoch,
                    train_metrics,
                    eval_metrics,
                    args,
                )

            if eval_metrics.loss < best_eval_loss:
                best_eval_loss = eval_metrics.loss
                best_eval_metrics = eval_metrics
                if args.save_best:
                    save_checkpoint(
                        models_dir,
                        model,
                        optimizer,
                        epoch,
                        train_metrics,
                        eval_metrics,
                        args,
                        best=True,
                    )

    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    if args.save_best and best_eval_metrics is not None:
        LOGGER.info(
            "Best eval loss %.4f (relative error %.2f%%)",
            best_eval_metrics.loss,
            best_eval_metrics.runtime_error * 100.0,
        )


if __name__ == "__main__":
    main()
