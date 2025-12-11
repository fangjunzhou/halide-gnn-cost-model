"""
PyTorch Dataset and DataLoader for pipeline DAGs.
"""

import logging
import json
from pathlib import Path
from typing import Callable, Optional

import networkx as nx
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData, Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from torch_geometric.utils import degree

try:
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - PyTorch < 2.6
    add_safe_globals = None

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


if add_safe_globals is not None:
    try:  # Allowlist PyG storage classes when using weights_only=True
        from torch_geometric.data.storage import (
            BaseStorage,
            EdgeStorage,
            GlobalStorage,
            NodeStorage,
        )

        add_safe_globals(
            [BaseStorage, EdgeStorage, GlobalStorage, NodeStorage, HeteroData]
        )
    except Exception as exc:  # pragma: no cover - best-effort registration
        logger.debug("Unable to register PyG storage classes for safe loading: %s", exc)


PREPROCESSED_FILENAME = "graph_data.pt"

_AST_VOCAB_SIZE: Optional[int] = None
_SCHED_VOCAB_SIZE: Optional[int] = None


def get_ast_vocab_size() -> int:
    global _AST_VOCAB_SIZE
    if _AST_VOCAB_SIZE is None:
        _AST_VOCAB_SIZE = len(build_ast_node_type_vocab())
    return _AST_VOCAB_SIZE


def get_sched_vocab_size() -> int:
    global _SCHED_VOCAB_SIZE
    if _SCHED_VOCAB_SIZE is None:
        _SCHED_VOCAB_SIZE = len(build_schedule_node_type_vocab())
    return _SCHED_VOCAB_SIZE


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
    dag_edges = list(dag.edges)
    edge_index = (
        torch.tensor(dag_edges, dtype=torch.int64).t().contiguous()
        if dag_edges
        else torch.empty((2, 0), dtype=torch.int64)
    )
    data["function", "called_by", "function"].edge_index = edge_index
    data["function", "call", "function"].edge_index = edge_index.flip(0).contiguous()

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
    ast_edges = list(ast_graph.edges)
    edge_index = (
        torch.tensor(ast_edges, dtype=torch.int64).t().contiguous()
        if ast_edges
        else torch.empty((2, 0), dtype=torch.int64)
    )
    data["ast_node", "child_of", "ast_node"].edge_index = edge_index
    data["ast_node", "parent_of", "ast_node"].edge_index = edge_index.flip(
        0
    ).contiguous()

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
        edge_index = torch.tensor(ast_to_func_edges, dtype=torch.int64).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64)
    data["ast_node", "is_expr_of", "function"].edge_index = edge_index
    data["function", "contains_expr", "ast_node"].edge_index = edge_index.flip(
        0
    ).contiguous()

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
    sched_edges = list(schedule_graph.edges)
    edge_index = (
        torch.tensor(sched_edges, dtype=torch.int64).t().contiguous()
        if sched_edges
        else torch.empty((2, 0), dtype=torch.int64)
    )
    data["loop_level", "child_of", "loop_level"].edge_index = edge_index
    data["loop_level", "parent_of", "loop_level"].edge_index = edge_index.flip(
        0
    ).contiguous()

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
        edge_index = (
            torch.tensor(func_to_sched_edges, dtype=torch.int64).t().contiguous()
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64)
    data["function", "schedule_at", "loop_level"].edge_index = edge_index
    data["loop_level", "schedule", "function"].edge_index = edge_index.flip(
        0
    ).contiguous()

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


def graph_transformer_preprocessor(
    data: HeteroData,
    *,
    k: int = 8,
    walk_length: int = 8,
    max_degree: int = 7,
) -> Data:
    """Project a heterogeneous pipeline graph to a homogeneous representation.

    Mirrors the experimental preprocessing pipeline used in
    ``notebooks/graph_transformer.ipynb`` by assigning node-type specific token
    IDs and computing positional encodings required by transformer backbones.

    :param data: Input pipeline graph as produced by ``load_pipeline`` or
        ``PipelineDataset``.
    :param k: Number of Laplacian eigenvectors to retain for positional
        encoding.
    :param walk_length: Random walk length for positional encodings.
    :param max_degree: Maximum node degree for one-hot encodings (values above
        this threshold are clamped).
    :return: Homogeneous ``torch_geometric.data.Data`` ready for transformer
        consumption.
    """

    if not isinstance(data, HeteroData):  # Defensive guard for easier debugging.
        raise TypeError("graph_transformer_preprocessor expects a HeteroData input")

    hetero = data.clone()

    # Ensure all node types expose integer tokens that survive homogenization.
    if "function" in hetero.node_types:
        func_store = hetero["function"]
        func_store.type = torch.zeros(
            (func_store.x.size(0), 1), dtype=torch.long, device=func_store.x.device
        )

    if "ast_node" in hetero.node_types:
        ast_store = hetero["ast_node"]
        ast_store.type = ast_store.x.clone().to(torch.long)

    if "loop_level" in hetero.node_types:
        loop_store = hetero["loop_level"]
        loop_store.type = loop_store.x.clone().to(torch.long)

    homogeneous = hetero.to_homogeneous(add_node_type=True, add_edge_type=True)

    ast_vocab_size = get_ast_vocab_size()
    sched_vocab_size = get_sched_vocab_size()

    node_tokens = homogeneous.type.view(-1).to(torch.long)
    node_type_lookup = {name: idx for idx, name in enumerate(hetero.node_types)}

    function_idx = node_type_lookup.get("function")
    ast_idx = node_type_lookup.get("ast_node")
    sched_idx = node_type_lookup.get("loop_level")

    if function_idx is not None:
        function_mask = homogeneous.node_type == function_idx
        node_tokens[function_mask] = 0

    if ast_idx is not None:
        ast_mask = homogeneous.node_type == ast_idx
        node_tokens[ast_mask] = node_tokens[ast_mask] + 1

    if sched_idx is not None:
        sched_mask = homogeneous.node_type == sched_idx
        node_tokens[sched_mask] = node_tokens[sched_mask] + ast_vocab_size + 1

    homogeneous.node_token = node_tokens
    homogeneous.x = node_tokens.unsqueeze(-1)

    num_nodes = homogeneous.num_nodes

    device = homogeneous.edge_index.device
    laplacian_pe = torch.zeros((num_nodes, k), dtype=torch.float, device=device)
    random_walk_pe = torch.zeros(
        (num_nodes, walk_length), dtype=torch.float, device=device
    )

    if num_nodes > 0:
        lap_k = min(k, num_nodes)
        if lap_k > 0:
            try:
                lap_data = AddLaplacianEigenvectorPE(k=lap_k)(homogeneous.clone())
                computed = lap_data.laplacian_eigenvector_pe
                laplacian_pe[:, : computed.size(-1)] = computed
            except RuntimeError:
                logger.debug(
                    "Failed to compute Laplacian PE; leaving zeros", exc_info=True
                )

        try:
            rw_data = AddRandomWalkPE(walk_length=walk_length)(homogeneous.clone())
            computed_rw = rw_data.random_walk_pe
            cols = min(random_walk_pe.size(-1), computed_rw.size(-1))
            if cols > 0:
                random_walk_pe[:, :cols] = computed_rw[:, :cols]
        except RuntimeError:
            logger.debug(
                "Failed to compute random walk PE; leaving zeros", exc_info=True
            )

    homogeneous.laplacian_eigenvector_pe = laplacian_pe
    homogeneous.random_walk_pe = random_walk_pe

    deg = degree(homogeneous.edge_index[0], num_nodes)
    deg = deg.clamp(max=max_degree).to(torch.long)
    homogeneous.degree_pe = F.one_hot(deg, num_classes=max_degree + 1).to(
        laplacian_pe.dtype
    )

    homogeneous.pe = torch.cat(
        (
            homogeneous.laplacian_eigenvector_pe,
            homogeneous.random_walk_pe,
            homogeneous.degree_pe,
        ),
        dim=-1,
    )

    homogeneous.total_token_vocab = ast_vocab_size + sched_vocab_size + 1

    return homogeneous


class PipelineDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        ast_vocab=None,
        sched_vocab=None,
        preload: bool = True,
        preprocessor: Optional[Callable[[HeteroData], Data | HeteroData]] = None,
    ) -> None:
        super().__init__()
        # Check if the directory exists
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise ValueError(
                f"Dataset directory {dataset_dir} does not exist or is not a directory."
            )
        # Get all the pipeline directories
        self.pipeline_dirs = sorted(
            [d for d in dataset_dir.iterdir() if d.is_dir()], key=lambda p: p.name
        )

        # Ensure preprocessed data is available for every pipeline directory.
        missing_preprocessed = [
            d for d in self.pipeline_dirs if not (d / PREPROCESSED_FILENAME).exists()
        ]
        if missing_preprocessed:
            missing_names = ", ".join(d.name for d in missing_preprocessed)
            raise FileNotFoundError(
                "Preprocessed graph data not found for pipelines: "
                f"{missing_names}. Please run pipepreprocess first."
            )

        # Build vocabs if not provided
        self.ast_vocab = (
            ast_vocab if ast_vocab is not None else build_ast_node_type_vocab()
        )
        self.sched_vocab = (
            sched_vocab if sched_vocab is not None else build_schedule_node_type_vocab()
        )

        self.preload = preload
        self.preprocessor = preprocessor
        self._data_cache = None
        if self.preload:
            # Eagerly load all preprocessed graphs to avoid repeated disk reads.
            self._data_cache = [
                self._apply_preprocessor(
                    self._load_preprocessed(d / PREPROCESSED_FILENAME)
                )
                for d in self.pipeline_dirs
            ]

    def __len__(self) -> int:
        return len(self.pipeline_dirs)

    def __getitem__(self, idx: int) -> HeteroData:
        if self.preload and self._data_cache is not None:
            return self._data_cache[idx]

        pipeline_dir = self.pipeline_dirs[idx]
        preprocessed_path = pipeline_dir / PREPROCESSED_FILENAME
        data = self._load_preprocessed(preprocessed_path)
        return self._apply_preprocessor(data)

    @staticmethod
    def _load_preprocessed(preprocessed_path: Path) -> HeteroData:
        load_kwargs = {"map_location": "cpu"}
        try:
            # Files are produced locally via `pipepreprocess`, so allowing full
            # object deserialization is acceptable here.
            data: HeteroData = torch.load(
                preprocessed_path, weights_only=False, **load_kwargs
            )
        except TypeError:  # pragma: no cover - PyTorch < 2.6
            data = torch.load(preprocessed_path, **load_kwargs)
        return data

    def _apply_preprocessor(self, data: HeteroData) -> Data | HeteroData:
        if self.preprocessor is None:
            return data

        processed = self.preprocessor(data.clone())
        if processed is None:
            raise ValueError("Preprocessor must return a processed graph instance")
        return processed
