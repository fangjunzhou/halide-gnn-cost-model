"""Schedule parser for Halide schedule trees.

Implements schedule node types and a visitor pattern for traversing
and analyzing schedule trees generated from Halide schedules.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import networkx as nx


class ScheduleVisitor:
    """Base class for schedule visitors.

    Implements the visitor pattern for traversing schedule nodes. Subclasses can override
    specific visit methods to implement custom behavior for different schedule node types.
    """

    def visit_root(self, node: ScheduleRoot):
        """Visit a root node.

        :param node: The root node to visit.
        """
        for child in node.children:
            child.accept(self)

    def visit_serial_for(self, node: SerialForNode):
        """Visit a serial for loop node.

        :param node: The serial for loop node to visit.
        """
        for child in node.body:
            child.accept(self)

    def visit_parallel_for(self, node: ParallelForNode):
        """Visit a parallel for loop node.

        :param node: The parallel for loop node to visit.
        """
        for child in node.body:
            child.accept(self)

    def visit_vectorized_for(self, node: VectorizedForNode):
        """Visit a vectorized for loop node.

        :param node: The vectorized for loop node to visit.
        """
        for child in node.body:
            child.accept(self)

    def visit_compute(self, node: ComputeNode):
        """Visit a compute node (leaf node).

        :param node: The compute node to visit.
        """
        return

    def visit_store(self, node: StoreNode):
        """Visit a store node.

        :param node: The store node to visit.
        """
        for child in node.body:
            child.accept(self)

    # Fallback dynamic dispatcher if visitor wants to call visit(node)
    def visit(self, node: ScheduleNode):
        """Dispatch to a visit_<kind> method based on node class name.

        This is a convenience method that dynamically dispatches to the appropriate
        visit method based on the node's class name.

        :param node: The schedule node to visit.
        :raises NotImplementedError: If no visit method exists for the node type.
        """
        name = type(node).__name__
        key = name.lower()
        method_name = f"visit_{key}"
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(f"Visitor has no method {method_name}")
        return method(node)


class ScheduleFormatVisitor(ScheduleVisitor):
    """Visitor that formats the schedule as a string representation.

    This visitor traverses the schedule tree and builds a string representation
    of the loop nest structure in a readable format.

    Example:
        visitor = ScheduleFormatVisitor()
        root_node.accept(visitor)
        formatted_string = visitor.get_result()
    """

    def __init__(self):
        """Initialize the visitor with an empty result and indentation level."""
        self.result = ""
        self.indent_level = 0

    def get_result(self) -> str:
        """Get the formatted string result.

        :return: The formatted schedule as a string.
        """
        return self.result

    def _indent(self) -> str:
        """Get the current indentation string.

        :return: A string of spaces for the current indentation level.
        """
        return "  " * self.indent_level

    def visit_root(self, node: ScheduleRoot):
        """Visit a root node and format all children.

        :param node: The root node to visit.
        """
        for child in node.children:
            child.accept(self)

    def visit_serial_for(self, node: SerialForNode):
        """Visit a serial for loop node and format it.

        :param node: The serial for loop node to visit.
        """
        self.result += (
            f"{self._indent()}for {node.var} [{node.min}:{node.max}] (serial)\n"
        )
        self.indent_level += 1
        for child in node.body:
            child.accept(self)
        self.indent_level -= 1

    def visit_parallel_for(self, node: ParallelForNode):
        """Visit a parallel for loop node and format it.

        :param node: The parallel for loop node to visit.
        """
        self.result += (
            f"{self._indent()}for {node.var} [{node.min}:{node.max}] (parallel)\n"
        )
        self.indent_level += 1
        for child in node.body:
            child.accept(self)
        self.indent_level -= 1

    def visit_vectorized_for(self, node: VectorizedForNode):
        """Visit a vectorized for loop node and format it.

        :param node: The vectorized for loop node to visit.
        """
        self.result += (
            f"{self._indent()}for {node.var} [{node.min}:{node.max}] (vectorized)\n"
        )
        self.indent_level += 1
        for child in node.body:
            child.accept(self)
        self.indent_level -= 1

    def visit_compute(self, node: ComputeNode):
        """Visit a compute node and format it as a compute statement.

        :param node: The compute node to visit.
        """
        self.result += f"{self._indent()}compute({node.func})\n"

    def visit_store(self, node: StoreNode):
        """Visit a store node and format it as a store statement with body.

        :param node: The store node to visit.
        """
        self.result += f"{self._indent()}store({node.func})\n"
        self.indent_level += 1
        for child in node.body:
            child.accept(self)
        self.indent_level -= 1


class ScheduleGraphVisitor(ScheduleVisitor):
    """Visitor that converts the schedule tree into a NetworkX directed graph.

    Creates a graph where each schedule node is represented as a graph node,
    and edges connect children to their parent nodes in the loop nest tree.
    Each graph node contains attributes about the schedule node type and properties.

    Example:
        import networkx as nx
        visitor = ScheduleGraphVisitor()
        root_node.accept(visitor)
        graph = visitor.get_graph()
    """

    def __init__(self):
        """Initialize the visitor with an empty graph and node counter."""
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.current_node_id = None
        self.node_stack = []  # Stack to track parent nodes

    def get_graph(self):
        """Get the constructed NetworkX graph.

        :return: A NetworkX DiGraph representing the schedule tree.
        """
        return self.graph

    def _create_node(self, node_type: str, **attributes) -> int:
        """Create a new graph node and return its ID.

        :param node_type: The type of the schedule node.
        :param attributes: Additional attributes to store in the node.
        :return: The ID of the created node.
        """
        node_id = self.node_counter
        self.node_counter += 1
        self.graph.add_node(node_id, node_type=node_type, **attributes)
        return node_id

    def _add_edge(self, child_id: int, parent_id: int):
        """Add an edge from child to parent.

        :param child_id: The ID of the child node.
        :param parent_id: The ID of the parent node.
        """
        self.graph.add_edge(child_id, parent_id)

    def visit_root(self, node: ScheduleRoot):
        """Visit a root node and create edges to children.

        :param node: The root node to visit.
        """
        node_id = self._create_node("Root")
        parent_id = self.node_stack[-1] if self.node_stack else None

        # Visit all children
        self.node_stack.append(node_id)
        child_ids = []
        for child in node.children:
            child.accept(self)
            child_ids.append(self.current_node_id)
        self.node_stack.pop()

        # Add edges from children to this node
        for child_id in child_ids:
            if child_id is not None:
                self._add_edge(child_id, node_id)

        self.current_node_id = node_id

    def visit_serial_for(self, node: SerialForNode):
        """Visit a serial for loop node and create edges to children.

        :param node: The serial for loop node to visit.
        """
        node_id = self._create_node(
            "SerialFor", var=node.var, min=node.min, max=node.max
        )
        parent_id = self.node_stack[-1] if self.node_stack else None

        # Visit all children
        self.node_stack.append(node_id)
        child_ids = []
        for child in node.body:
            child.accept(self)
            child_ids.append(self.current_node_id)
        self.node_stack.pop()

        # Add edges from children to this node
        for child_id in child_ids:
            if child_id is not None:
                self._add_edge(child_id, node_id)

        self.current_node_id = node_id

    def visit_parallel_for(self, node: ParallelForNode):
        """Visit a parallel for loop node and create edges to children.

        :param node: The parallel for loop node to visit.
        """
        node_id = self._create_node(
            "ParallelFor", var=node.var, min=node.min, max=node.max
        )
        parent_id = self.node_stack[-1] if self.node_stack else None

        # Visit all children
        self.node_stack.append(node_id)
        child_ids = []
        for child in node.body:
            child.accept(self)
            child_ids.append(self.current_node_id)
        self.node_stack.pop()

        # Add edges from children to this node
        for child_id in child_ids:
            if child_id is not None:
                self._add_edge(child_id, node_id)

        self.current_node_id = node_id

    def visit_vectorized_for(self, node: VectorizedForNode):
        """Visit a vectorized for loop node and create edges to children.

        :param node: The vectorized for loop node to visit.
        """
        node_id = self._create_node(
            "VectorizedFor", var=node.var, min=node.min, max=node.max
        )
        parent_id = self.node_stack[-1] if self.node_stack else None

        # Visit all children
        self.node_stack.append(node_id)
        child_ids = []
        for child in node.body:
            child.accept(self)
            child_ids.append(self.current_node_id)
        self.node_stack.pop()

        # Add edges from children to this node
        for child_id in child_ids:
            if child_id is not None:
                self._add_edge(child_id, node_id)

        self.current_node_id = node_id

    def visit_compute(self, node: ComputeNode):
        """Visit a compute node (leaf node).

        :param node: The compute node to visit.
        """
        node_id = self._create_node("Compute", func=node.func)
        self.current_node_id = node_id

    def visit_store(self, node: StoreNode):
        """Visit a store node and create edges to children.

        :param node: The store node to visit.
        """
        node_id = self._create_node("Store", func=node.func)
        parent_id = self.node_stack[-1] if self.node_stack else None

        # Visit all children
        self.node_stack.append(node_id)
        child_ids = []
        for child in node.body:
            child.accept(self)
            child_ids.append(self.current_node_id)
        self.node_stack.pop()

        # Add edges from children to this node
        for child_id in child_ids:
            if child_id is not None:
                self._add_edge(child_id, node_id)

        self.current_node_id = node_id


class ScheduleNode(ABC):
    """Abstract base class for all schedule nodes.

    All concrete schedule node types must inherit from this class and implement
    the accept method to support the visitor pattern.
    """

    @abstractmethod
    def accept(self, visitor: ScheduleVisitor):
        """Accept a visitor and dispatch to the appropriate visit method.

        :param visitor: The visitor to accept.
        """
        pass


class ForNode(ScheduleNode):
    """Base class for all for loop nodes in the schedule.

    :param var: The loop variable name (e.g., "v0", "v0.v0_in").
    :param body: List of child schedule nodes in the loop body.
    :param min: Optional minimum bound of the loop (as string).
    :param max: Optional maximum bound of the loop (as string).
    """

    def __init__(
        self,
        var: str,
        body: List[ScheduleNode],
        min: str | None = None,
        max: str | None = None,
    ):
        self.var = var
        self.body = body
        self.min = min if min is not None else "0"
        self.max = max if max is not None else "unknown"

    @abstractmethod
    def accept(self, visitor: ScheduleVisitor):
        """Accept a visitor and dispatch to the appropriate visit method.

        :param visitor: The visitor to accept.
        """
        pass


class SerialForNode(ForNode):
    """Node representing a serial for loop in the schedule."""

    def accept(self, visitor: ScheduleVisitor):
        visitor.visit_serial_for(self)


class ParallelForNode(ForNode):
    """Node representing a parallel for loop in the schedule."""

    def accept(self, visitor: ScheduleVisitor):
        visitor.visit_parallel_for(self)


class VectorizedForNode(ForNode):
    """Node representing a vectorized for loop in the schedule."""

    def accept(self, visitor: ScheduleVisitor):
        visitor.visit_vectorized_for(self)


class ComputeNode(ScheduleNode):
    """Node representing a compute statement in the schedule.

    Represents where a function is computed in the schedule tree.

    :param func: The name of the function being computed.
    """

    def __init__(self, func: str):
        self.func = func

    def accept(self, visitor: ScheduleVisitor):
        visitor.visit_compute(self)


class StoreNode(ScheduleNode):
    """Node representing a store statement in the schedule.

    Represents where intermediate results for a function are stored and includes
    the nested loop structure and computation nodes in its body.

    :param func: The name of the function being stored.
    :param body: List of child schedule nodes in the store body.
    """

    def __init__(self, func: str, body: List[ScheduleNode]):
        self.func = func
        self.body = body

    def accept(self, visitor: ScheduleVisitor):
        visitor.visit_store(self)


class ScheduleRoot(ScheduleNode):
    """Root of a schedule loop nest tree.

    Represents the top-level container for multiple schedule nodes (typically
    a mix of For loops and Store statements).

    :param children: List of top-level schedule nodes.
    """

    def __init__(self, children: List[ScheduleNode]):
        self.children = children

    def accept(self, visitor: ScheduleVisitor):
        """Accept a visitor for all children nodes.

        :param visitor: The visitor to accept.
        """
        visitor.visit_root(self)


def parse_schedule(schedule_json_obj) -> ScheduleRoot:
    """Parse a schedule JSON object into a schedule tree.

    :param schedule_json_obj: The schedule represented as a JSON object (list of nodes).
    :return: A ScheduleRoot object representing the parsed schedule tree.
    """

    def parse_node(node_dict) -> ScheduleNode:
        """Recursively parse a JSON node into a ScheduleNode.

        :param node_dict: Dictionary representing a single schedule node.
        :return: A ScheduleNode object.
        """
        node_type = node_dict.get("type")

        if node_type == "For":
            var = node_dict["var"]
            for_type = node_dict["for_type"]
            min_bound = node_dict.get("min")
            max_bound = node_dict.get("max")
            body = [parse_node(child) for child in node_dict.get("body", [])]

            if for_type == "Serial":
                return SerialForNode(var, body, min_bound, max_bound)
            elif for_type == "Parallel":
                return ParallelForNode(var, body, min_bound, max_bound)
            elif for_type == "Vectorized":
                return VectorizedForNode(var, body, min_bound, max_bound)
            else:
                raise ValueError(f"Unknown for loop type: {for_type}")

        elif node_type == "Compute":
            func = node_dict["func"]
            return ComputeNode(func)

        elif node_type == "Store":
            func = node_dict["func"]
            body = [parse_node(child) for child in node_dict.get("body", [])]
            return StoreNode(func, body)

        else:
            raise ValueError(f"Unknown schedule node type: {node_type}")

    # Parse the top-level list of schedule nodes
    root_nodes = []
    for node_dict in schedule_json_obj:
        root_nodes.append(parse_node(node_dict))

    return ScheduleRoot(root_nodes)
