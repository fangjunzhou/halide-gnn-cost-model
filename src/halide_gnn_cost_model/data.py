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

from halide_gnn_cost_model.ast_parser import (
    parse_ast,
    ASTGraphVisitor,
    build_ast_node_type_vocab,
)
from halide_gnn_cost_model.schedule_parser import (
    parse_schedule,
    ScheduleGraphVisitor,
    build_schedule_node_type_vocab,
)


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


def load_ast_graph(ast_path: Path) -> nx.DiGraph:
    """Load an AST graph from a JSON file.

    :param ast_path: Path to the AST JSON file.
    :return: A NetworkX DiGraph representing the AST.
    """
    with open(ast_path, "r") as f:
        ast_data = json.load(f)
    ast_roots = parse_ast(ast_data)
    ast_graphs = []
    for root in ast_roots:
        visitor = ASTGraphVisitor()
        root.accept(visitor)
        ast_graphs.append(visitor.get_graph())
    ast_graph = nx.disjoint_union_all(ast_graphs)
    return ast_graph


def load_schedule_graph(schedule_path: Path) -> nx.DiGraph:
    """Load a schedule graph from a JSON file.

    :param schedule_path: Path to the schedule JSON file.
    :return: A NetworkX DiGraph representing the schedule.
    """
    with open(schedule_path, "r") as f:
        schedule_data = json.load(f)
    schedule_root = parse_schedule(schedule_data)
    graph_visitor = ScheduleGraphVisitor()
    schedule_root.accept(graph_visitor)
    schedule_graph = graph_visitor.get_graph()
    return schedule_graph


def load_pipeline(pipeline_dir: Path, ast_vocab=None, sched_vocab=None) -> HeteroData:
    """Load a pipeline DAG from the specified directory.

    :param pipeline_dir: Path to the pipeline directory containing AST, DAG, and schedule json files.
    :param ast_vocab: Vocabulary for AST node types. If None, will be built from scratch.
    :param sched_vocab: Vocabulary for schedule node types. If None, will be built from scratch.
    :return: A PyTorch Geometric Data object representing the pipeline.
    """
    data = HeteroData()

    # Build vocabs if not provided
    if ast_vocab is None:
        ast_vocab = build_ast_node_type_vocab()
    if sched_vocab is None:
        sched_vocab = build_schedule_node_type_vocab()

    # --------------------------- DAG  --------------------------- #

    # Load the DAG JSON file.
    dag_path = pipeline_dir / "dag.json"
    dag = load_dag(dag_path)
    data["function"].x = torch.ones((len(dag.nodes), 1), dtype=torch.float)
    edge_index = torch.tensor(list(dag.edges)).T
    data["function", "called_by", "function"].edge_index = edge_index

    # Create function name to DAG node ID mapping
    func_names = [dag.nodes[n].get("name", {""}) for n in dag.nodes()]
    func_names = [
        next(iter(name_set)) if isinstance(name_set, set) else name_set
        for name_set in func_names
    ]
    func_name_to_id = {name: idx for idx, name in enumerate(func_names)}

    # ------------------------ AST Nodes  ------------------------ #

    ast_path = pipeline_dir / "ast.json"
    ast_graph = load_ast_graph(ast_path)

    # Add AST nodes
    node_types = [ast_node["node_type"] for _, ast_node in ast_graph.nodes(data=True)]
    node_tokens = [ast_vocab[node_type] for node_type in node_types]
    data["ast_node"].x = torch.tensor(node_tokens, dtype=torch.int).unsqueeze(-1)

    # Add AST node-to-node edges
    if list(ast_graph.edges):
        edge_index = torch.tensor(list(ast_graph.edges)).T.contiguous()
        data["ast_node", "child_of", "ast_node"].edge_index = edge_index

    # Connect AST to corresponding function nodes
    ast_to_func_edges = []
    for ast_node_idx, ast_node in ast_graph.nodes(data=True):
        if ast_node["node_type"] != "Root":
            continue
        func_name = ast_node.get("function")
        if func_name and func_name in func_name_to_id:
            func_idx = func_name_to_id[func_name]
            ast_to_func_edges.append((ast_node_idx, func_idx))
    if ast_to_func_edges:
        edge_index = torch.tensor(ast_to_func_edges).T.contiguous()
        data["ast_node", "is_expr_of", "function"].edge_index = edge_index

    # ---------------- Schedule/Loop Level Nodes  ---------------- #

    schedule_path = pipeline_dir / "schedule.json"
    schedule_graph = load_schedule_graph(schedule_path)

    # Add loop level nodes
    node_types = [
        sched_node["node_type"] for _, sched_node in schedule_graph.nodes(data=True)
    ]
    node_tokens = [sched_vocab[node_type] for node_type in node_types]
    data["loop_level"].x = torch.tensor(node_tokens, dtype=torch.int).unsqueeze(-1)

    # Add loop level-to-loop level edges
    if list(schedule_graph.edges):
        edge_index = torch.tensor(list(schedule_graph.edges)).T.contiguous()
        data["loop_level", "child_of", "loop_level"].edge_index = edge_index

    # Create function to schedule loop level mapping
    func_to_loop_levels = {}
    for idx, loop_level in schedule_graph.nodes(data=True):
        if loop_level["node_type"] in ("Compute", "Store"):
            func_name = loop_level.get("func")
            if func_name:
                if func_name not in func_to_loop_levels:
                    func_to_loop_levels[func_name] = []
                func_to_loop_levels[func_name].append(idx)

    # Connect function to corresponding schedule nodes
    func_to_sched_edges = []
    for func_idx in range(len(dag.nodes)):
        func_name = func_names[func_idx]
        if func_name in func_to_loop_levels:
            for sched_node_idx in func_to_loop_levels[func_name]:
                func_to_sched_edges.append((func_idx, sched_node_idx))
    if func_to_sched_edges:
        edge_index = torch.tensor(func_to_sched_edges).T.contiguous()
        data["function", "schedule_at", "loop_level"].edge_index = edge_index

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
    def __init__(self, dataset_dir: Path, ast_vocab=None, sched_vocab=None) -> None:
        super().__init__()
        # Check if the directory exists
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise ValueError(
                f"Dataset directory {dataset_dir} does not exist or is not a directory."
            )
        # Get all the pipeline directories
        self.pipeline_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

        # Build vocabs if not provided
        self.ast_vocab = (
            ast_vocab if ast_vocab is not None else build_ast_node_type_vocab()
        )
        self.sched_vocab = (
            sched_vocab if sched_vocab is not None else build_schedule_node_type_vocab()
        )

    def __len__(self) -> int:
        return len(self.pipeline_dirs)

    def __getitem__(self, idx: int) -> HeteroData:
        pipeline_dir = self.pipeline_dirs[idx]
        data = load_pipeline(pipeline_dir, self.ast_vocab, self.sched_vocab)
        return data
