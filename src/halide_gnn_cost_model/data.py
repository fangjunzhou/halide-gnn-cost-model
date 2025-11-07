"""
PyTorch Dataset and DataLoader for pipeline DAGs.
"""

import logging
import torch
from torch.utils.data import Dataset
from pathlib import Path
import networkx as nx
import json
from torch_geometric.data import HeteroData


logger = logging.getLogger(__name__)


def load_dag(dag_path: Path) -> nx.DiGraph:
    """Load a DAG from a JSON file.

    :param dag_path: Path to the DAG JSON file.
    :return: A NetworkX DiGraph representing the DAG.
    """
    with open(dag_path, "r") as f:
        dag_data = json.load(f)
    func_names = [func["name"] for func in dag_data]
    dag = nx.DiGraph()
    # Add function nodes.
    for idx, func_name in enumerate(func_names):
        dag.add_node(idx, name={func_name})
    # Add edges based on dependencies.
    for idx, func in enumerate(dag_data):
        for dep in func["parents"]:
            dep_idx = func_names.index(dep)
            dag.add_edge(dep_idx, idx)
    return dag


def load_pipeline(pipeline_dir: Path) -> HeteroData:
    """Load a pipeline DAG from the specified directory.

    :param pipeline_dir: Path to the pipeline directory containing AST, DAG, and schedule json files.
    :return: A PyTorch Geometric Data object representing the pipeline.
    """
    data = HeteroData()

    # --------------------------- DAG  --------------------------- #

    # Load the DAG JSON file.
    dag_path = pipeline_dir / "dag.json"
    dag = load_dag(dag_path)
    data["function"].x = torch.ones((len(dag.nodes), 1), dtype=torch.float)
    edge_index = torch.tensor(list(dag.edges)).T
    data["function", "called_by", "function"].edge_index = edge_index

    # --------------------- Benchmark Label  --------------------- #

    benchmark_path = pipeline_dir / "benchmark.json"
    with open(benchmark_path, "r") as f:
        benchmark_data = json.load(f)
    y = torch.tensor(
        [benchmark["real_time"] for benchmark in benchmark_data["benchmarks"]],
        dtype=torch.float,
    )
    data.y = y
    return data


class PipelineDataset(Dataset):
    def __init__(self, dataset_dir: Path) -> None:
        super().__init__()
        # Check if the directory exists
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise ValueError(
                f"Dataset directory {dataset_dir} does not exist or is not a directory."
            )
        # Get all the pipeline directories
        self.pipeline_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

    def __len__(self) -> int:
        return len(self.pipeline_dirs)

    def __getitem__(self, idx: int) -> HeteroData:
        pipeline_dir = self.pipeline_dirs[idx]
        data = load_pipeline(pipeline_dir)
        return data
