from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class ASTVisitor:
    """Base class for AST visitors.

    Implements the visitor pattern for traversing AST nodes. Subclasses can override
    specific visit methods to implement custom behavior for different node types.
    """

    def visit_root(self, node: RootNode):
        """Visit a root node.

        :param node: The root node to visit.
        """
        node.child.accept(self)

    def visit_add(self, node: AddNode):
        """Visit an addition node.

        :param node: The addition node to visit.
        """
        node.left.accept(self)
        node.right.accept(self)

    def visit_sub(self, node: SubNode):
        """Visit a subtraction node.

        :param node: The subtraction node to visit.
        """
        node.left.accept(self)
        node.right.accept(self)

    def visit_mul(self, node: MulNode):
        """Visit a multiplication node.

        :param node: The multiplication node to visit.
        """
        node.left.accept(self)
        node.right.accept(self)

    def visit_div(self, node: DivNode):
        """Visit a division node.

        :param node: The division node to visit.
        """
        node.left.accept(self)
        node.right.accept(self)

    def visit_variable(self, node: VariableNode):
        """Visit a variable node (leaf node).

        :param node: The variable node to visit.
        """
        return

    def visit_intimm(self, node: IntImmNode):
        """Visit an integer immediate node (leaf node).

        :param node: The integer immediate node to visit.
        """
        return

    def visit_floatimm(self, node: FloatImmNode):
        """Visit a floating-point immediate node (leaf node).

        :param node: The floating-point immediate node to visit.
        """
        return

    def visit_call(self, node: CallNode):
        """Visit a function call node.

        :param node: The call node to visit.
        """
        for a in node.args:
            a.accept(self)

    def visit_cast(self, node: CastNode):
        """Visit a cast node.

        :param node: The cast node to visit.
        """
        node.value.accept(self)

    def visit_ramp(self, node: RampNode):
        """Visit a ramp node (vectorization primitive).

        :param node: The ramp node to visit.
        """
        node.base.accept(self)
        node.stride.accept(self)

    def visit_broadcast(self, node: BroadcastNode):
        """Visit a broadcast node (vectorization primitive).

        :param node: The broadcast node to visit.
        """
        node.value.accept(self)

    # Fallback dynamic dispatcher if visitor wants to call visit(node)
    def visit(self, node: ASTNode):
        """Dispatch to a visit_<kind> method based on node class name.

        This is a convenience method that dynamically dispatches to the appropriate
        visit method based on the node's class name. For example, an AddNode will
        be dispatched to visit_add.

        :param node: The AST node to visit.
        :raises NotImplementedError: If no visit method exists for the node type.

        Example:
            visitor = ASTVisitor()
            visitor.visit(AddNode(...))  # Calls visit_add
        """
        name = type(node).__name__
        key = name.lower()
        method_name = f"visit_{key}"
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(f"Visitor has no method {method_name}")
        return method(node)


class ASTFormatVisitor(ASTVisitor):
    """Visitor that formats the AST as a string expression.

    This visitor traverses the AST and builds a string representation
    of the expression in a readable format.

    Example:
        visitor = ASTFormatVisitor()
        root_node.accept(visitor)
        formatted_string = visitor.get_result()
    """

    def __init__(self):
        """Initialize the visitor with an empty result."""
        self.result = ""

    def get_result(self) -> str:
        """Get the formatted string result.

        :return: The formatted AST as a string.
        """
        return self.result

    def visit_root(self, node: RootNode):
        """Visit a root node and format it with the function name.

        :param node: The root node to visit.
        """
        self.result = f"{node.function} = "
        node.child.accept(self)

    def visit_add(self, node: AddNode):
        """Visit an addition node and format it as (left + right).

        :param node: The addition node to visit.
        """
        self.result += "("
        node.left.accept(self)
        self.result += " + "
        node.right.accept(self)
        self.result += ")"

    def visit_sub(self, node: SubNode):
        """Visit a subtraction node and format it as (left - right).

        :param node: The subtraction node to visit.
        """
        self.result += "("
        node.left.accept(self)
        self.result += " - "
        node.right.accept(self)
        self.result += ")"

    def visit_mul(self, node: MulNode):
        """Visit a multiplication node and format it as (left * right).

        :param node: The multiplication node to visit.
        """
        self.result += "("
        node.left.accept(self)
        self.result += " * "
        node.right.accept(self)
        self.result += ")"

    def visit_div(self, node: DivNode):
        """Visit a division node and format it as (left / right).

        :param node: The division node to visit.
        """
        self.result += "("
        node.left.accept(self)
        self.result += " / "
        node.right.accept(self)
        self.result += ")"

    def visit_variable(self, node: VariableNode):
        """Visit a variable node and append its name.

        :param node: The variable node to visit.
        """
        self.result += node.name

    def visit_intimm(self, node: IntImmNode):
        """Visit an integer immediate node and append its value.

        :param node: The integer immediate node to visit.
        """
        self.result += str(node.value)

    def visit_floatimm(self, node: FloatImmNode):
        """Visit a floating-point immediate node and append its value.

        :param node: The floating-point immediate node to visit.
        """
        self.result += str(node.value)

    def visit_call(self, node: CallNode):
        """Visit a function call node and format it as name(arg1, arg2, ...).

        :param node: The call node to visit.
        """
        self.result += f"{node.name}("
        for i, arg in enumerate(node.args):
            if i > 0:
                self.result += ", "
            arg.accept(self)
        self.result += ")"

    def visit_cast(self, node: CastNode):
        """Visit a cast node and format it as cast<type>(value).

        :param node: The cast node to visit.
        """
        if node.type_name:
            self.result += f"cast<{node.type_name}>("
        else:
            self.result += "cast("
        node.value.accept(self)
        self.result += ")"

    def visit_ramp(self, node: RampNode):
        """Visit a ramp node and format it as ramp(base, stride, lanes).

        :param node: The ramp node to visit.
        """
        self.result += "ramp("
        node.base.accept(self)
        self.result += ", "
        node.stride.accept(self)
        self.result += f", {node.lanes})"

    def visit_broadcast(self, node: BroadcastNode):
        """Visit a broadcast node and format it as broadcast(value, lanes).

        :param node: The broadcast node to visit.
        """
        self.result += "broadcast("
        node.value.accept(self)
        self.result += f", {node.lanes})"


class ASTGraphVisitor(ASTVisitor):
    """Visitor that converts the AST into a NetworkX directed graph.

    Creates a graph where each AST node is represented as a graph node,
    and edges connect children to their parent nodes in the expression tree.
    Each graph node contains attributes about the AST node type and values.

    Example:
        import networkx as nx
        visitor = ASTGraphVisitor()
        root_node.accept(visitor)
        graph = visitor.get_graph()
    """

    def __init__(self):
        """Initialize the visitor with an empty graph and node counter."""
        import networkx as nx

        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.current_node_id = None
        self.node_stack = []  # Stack to track parent nodes

    def get_graph(self):
        """Get the constructed NetworkX graph.

        :return: A NetworkX DiGraph representing the AST.
        """
        return self.graph

    def _create_node(self, node_type: str, **attributes) -> int:
        """Create a new graph node and return its ID.

        :param node_type: The type of the AST node.
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

    def visit_root(self, node: RootNode):
        """Visit a root node and create a root graph node.

        :param node: The root node to visit.
        """
        root_id = self._create_node("Root", function=node.function)
        self.node_stack.append(root_id)
        node.child.accept(self)
        # Connect child to root using current_node_id
        child_id = self.current_node_id
        if child_id is not None and child_id != root_id:
            self._add_edge(child_id, root_id)
        self.node_stack.pop()
        self.current_node_id = root_id

    def visit_add(self, node: AddNode):
        """Visit an addition node and create edges to children.

        :param node: The addition node to visit.
        """
        node_id = self._create_node("Add", operator="+")
        parent_id = self.node_stack[-1] if self.node_stack else None

        # Visit left child
        self.node_stack.append(node_id)
        node.left.accept(self)
        left_id = self.current_node_id

        # Visit right child
        node.right.accept(self)
        right_id = self.current_node_id
        self.node_stack.pop()

        # Add edges from children to this node
        if left_id is not None:
            self._add_edge(left_id, node_id)
        if right_id is not None:
            self._add_edge(right_id, node_id)

        self.current_node_id = node_id

    def visit_sub(self, node: SubNode):
        """Visit a subtraction node and create edges to children.

        :param node: The subtraction node to visit.
        """
        node_id = self._create_node("Sub", operator="-")
        parent_id = self.node_stack[-1] if self.node_stack else None

        # Visit left child
        self.node_stack.append(node_id)
        node.left.accept(self)
        left_id = self.current_node_id

        # Visit right child
        node.right.accept(self)
        right_id = self.current_node_id
        self.node_stack.pop()

        # Add edges from children to this node
        if left_id is not None:
            self._add_edge(left_id, node_id)
        if right_id is not None:
            self._add_edge(right_id, node_id)

        self.current_node_id = node_id

    def visit_mul(self, node: MulNode):
        """Visit a multiplication node and create edges to children.

        :param node: The multiplication node to visit.
        """
        node_id = self._create_node("Mul", operator="*")
        parent_id = self.node_stack[-1] if self.node_stack else None

        # Visit left child
        self.node_stack.append(node_id)
        node.left.accept(self)
        left_id = self.current_node_id

        # Visit right child
        node.right.accept(self)
        right_id = self.current_node_id
        self.node_stack.pop()

        # Add edges from children to this node
        if left_id is not None:
            self._add_edge(left_id, node_id)
        if right_id is not None:
            self._add_edge(right_id, node_id)

        self.current_node_id = node_id

    def visit_div(self, node: DivNode):
        """Visit a division node and create edges to children.

        :param node: The division node to visit.
        """
        node_id = self._create_node("Div", operator="/")
        parent_id = self.node_stack[-1] if self.node_stack else None

        # Visit left child
        self.node_stack.append(node_id)
        node.left.accept(self)
        left_id = self.current_node_id

        # Visit right child
        node.right.accept(self)
        right_id = self.current_node_id
        self.node_stack.pop()

        # Add edges from children to this node
        if left_id is not None:
            self._add_edge(left_id, node_id)
        if right_id is not None:
            self._add_edge(right_id, node_id)

        self.current_node_id = node_id

    def visit_variable(self, node: VariableNode):
        """Visit a variable node (leaf node).

        :param node: The variable node to visit.
        """
        node_id = self._create_node("Variable", variable=node.name)
        self.current_node_id = node_id

    def visit_intimm(self, node: IntImmNode):
        """Visit an integer immediate node (leaf node).

        :param node: The integer immediate node to visit.
        """
        node_id = self._create_node("IntImm", value=node.value)
        self.current_node_id = node_id

    def visit_floatimm(self, node: FloatImmNode):
        """Visit a floating-point immediate node (leaf node).

        :param node: The floating-point immediate node to visit.
        """
        node_id = self._create_node("FloatImm", value=node.value)
        self.current_node_id = node_id

    def visit_call(self, node: CallNode):
        """Visit a function call node and create edges to argument nodes.

        :param node: The call node to visit.
        """
        node_id = self._create_node("Call", function=node.name)

        # Visit all arguments
        self.node_stack.append(node_id)
        arg_ids = []
        for arg in node.args:
            arg.accept(self)
            arg_ids.append(self.current_node_id)
        self.node_stack.pop()

        # Add edges from arguments to this call node
        for arg_id in arg_ids:
            if arg_id is not None:
                self._add_edge(arg_id, node_id)

        self.current_node_id = node_id

    def visit_cast(self, node: CastNode):
        """Visit a cast node and create edge to the value.

        :param node: The cast node to visit.
        """
        node_id = self._create_node("Cast", type_name=node.type_name)

        # Visit the value being cast
        self.node_stack.append(node_id)
        node.value.accept(self)
        value_id = self.current_node_id
        self.node_stack.pop()

        # Add edge from value to this cast node
        if value_id is not None:
            self._add_edge(value_id, node_id)

        self.current_node_id = node_id

    def visit_ramp(self, node: RampNode):
        """Visit a ramp node and create edges to base and stride.

        :param node: The ramp node to visit.
        """
        node_id = self._create_node("Ramp", lanes=node.lanes)

        # Visit base and stride
        self.node_stack.append(node_id)
        node.base.accept(self)
        base_id = self.current_node_id

        node.stride.accept(self)
        stride_id = self.current_node_id
        self.node_stack.pop()

        # Add edges from base and stride to this ramp node
        if base_id is not None:
            self._add_edge(base_id, node_id)
        if stride_id is not None:
            self._add_edge(stride_id, node_id)

        self.current_node_id = node_id

    def visit_broadcast(self, node: BroadcastNode):
        """Visit a broadcast node and create edge to the value.

        :param node: The broadcast node to visit.
        """
        node_id = self._create_node("Broadcast", lanes=node.lanes)

        # Visit the value being broadcast
        self.node_stack.append(node_id)
        node.value.accept(self)
        value_id = self.current_node_id
        self.node_stack.pop()

        # Add edge from value to this broadcast node
        if value_id is not None:
            self._add_edge(value_id, node_id)

        self.current_node_id = node_id


class ASTNode(ABC):
    """Abstract base class for all AST nodes.

    All concrete AST node types must inherit from this class and implement
    the accept method to support the visitor pattern.
    """

    @abstractmethod
    def accept(self, visitor: ASTVisitor):
        """Accept a visitor and dispatch to the appropriate visit method.

        :param visitor: The visitor to accept.
        """
        pass


class RootNode(ASTNode):
    """Root node of the AST for a function.

    :param function: Name of the function.
    :param child: Child AST node representing the function body.
    """

    def __init__(self, function: str, child: ASTNode):
        self.function = function
        self.child = child

    def accept(self, visitor: ASTVisitor):
        visitor.visit_root(self)


class AddNode(ASTNode):
    """Node representing an addition operation.

    :param left: Left operand AST node.
    :param right: Right operand AST node.
    """

    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def accept(self, visitor: ASTVisitor):
        visitor.visit_add(self)


class SubNode(ASTNode):
    """Node representing a subtraction operation.

    :param left: Left operand AST node.
    :param right: Right operand AST node.
    """

    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def accept(self, visitor: ASTVisitor):
        visitor.visit_sub(self)


class MulNode(ASTNode):
    """Node representing a multiplication operation.

    :param left: Left operand AST node.
    :param right: Right operand AST node.
    """

    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def accept(self, visitor: ASTVisitor):
        visitor.visit_mul(self)


class DivNode(ASTNode):
    """Node representing a division operation.

    :param left: Left operand AST node.
    :param right: Right operand AST node.
    """

    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def accept(self, visitor: ASTVisitor):
        visitor.visit_div(self)


class VariableNode(ASTNode):
    """Variable leaf node.

    Represents a named variable reference in the AST.

    :param name: The name of the variable.
    """

    def __init__(self, name: str):
        self.name = name

    def accept(self, visitor: ASTVisitor):
        visitor.visit_variable(self)


class IntImmNode(ASTNode):
    """Integer immediate (constant).

    Represents a constant integer value in the AST.

    :param value: The integer value.
    """

    def __init__(self, value: int):
        self.value = value

    def accept(self, visitor: ASTVisitor):
        visitor.visit_intimm(self)


class FloatImmNode(ASTNode):
    """Floating-point immediate (constant).

    Represents a constant floating-point value in the AST.

    :param value: The floating-point value.
    """

    def __init__(self, value: float):
        self.value = value

    def accept(self, visitor: ASTVisitor):
        visitor.visit_floatimm(self)


class CallNode(ASTNode):
    """Function/call node with a name and list of argument ASTNodes.

    Represents a function call in the AST.

    :param name: The name of the function being called.
    :param args: List of AST nodes representing the function arguments.
    """

    def __init__(self, name: str, args: list[ASTNode]):
        self.name = name
        self.args = args

    def accept(self, visitor: ASTVisitor):
        visitor.visit_call(self)


class CastNode(ASTNode):
    """Cast node wrapping a value. Type information kept as a string.

    Represents a type cast operation in the AST.

    :param value: The AST node being cast.
    :param type_name: Optional string describing the target type (e.g., "code:0,bits:32,lanes:1").
    """

    def __init__(self, value: ASTNode, type_name: str | None = None):
        self.value = value
        self.type_name = type_name

    def accept(self, visitor: ASTVisitor):
        visitor.visit_cast(self)


class RampNode(ASTNode):
    """Vector ramp node: base + stride * i for lanes.

    Represents a vectorization primitive that generates a sequence of values.
    Used in SIMD operations where each lane gets base + stride * lane_index.

    :param base: The base value AST node.
    :param stride: The stride value AST node.
    :param lanes: The number of vector lanes.
    """

    def __init__(self, base: ASTNode, stride: ASTNode, lanes: int):
        self.base = base
        self.stride = stride
        self.lanes = lanes

    def accept(self, visitor: ASTVisitor):
        visitor.visit_ramp(self)


class BroadcastNode(ASTNode):
    """Broadcast a scalar value to multiple lanes.

    Represents a vectorization primitive that replicates a scalar value across
    multiple vector lanes for SIMD operations.

    :param value: The scalar AST node to broadcast.
    :param lanes: The number of vector lanes to broadcast to.
    """

    def __init__(self, value: ASTNode, lanes: int):
        self.value = value
        self.lanes = lanes

    def accept(self, visitor: ASTVisitor):
        visitor.visit_broadcast(self)


def parse_ast(ast_json_obj) -> List[RootNode]:
    """Parse the AST JSON object and return a list of RootNode objects.

    :param ast_json_obj: The AST represented as a JSON object.
    :return: A list of RootNode objects representing the AST.
    """

    def parse_node(node_dict) -> ASTNode:
        """Recursively parse a JSON node into an ASTNode.

        :param node_dict: Dictionary representing a single AST node.
        :return: An ASTNode object.
        """
        node_type = node_dict.get("type")

        if node_type == "Variable":
            return VariableNode(node_dict["name"])

        elif node_type == "IntImm":
            return IntImmNode(node_dict["value"])

        elif node_type == "FloatImm":
            return FloatImmNode(node_dict["value"])

        elif node_type == "Add":
            left = parse_node(node_dict["left"])
            right = parse_node(node_dict["right"])
            return AddNode(left, right)

        elif node_type == "Sub":
            left = parse_node(node_dict["left"])
            right = parse_node(node_dict["right"])
            return SubNode(left, right)

        elif node_type == "Mul":
            left = parse_node(node_dict["left"])
            right = parse_node(node_dict["right"])
            return MulNode(left, right)

        elif node_type == "Div":
            left = parse_node(node_dict["left"])
            right = parse_node(node_dict["right"])
            return DivNode(left, right)

        elif node_type == "Call":
            name = node_dict["name"]
            args = [parse_node(arg) for arg in node_dict.get("args", [])]
            return CallNode(name, args)

        elif node_type == "Cast":
            value = parse_node(node_dict["value"])
            type_name = node_dict.get("type_name")
            return CastNode(value, type_name)

        elif node_type == "Ramp":
            base = parse_node(node_dict["base"])
            stride = parse_node(node_dict["stride"])
            lanes = node_dict["lanes"]
            return RampNode(base, stride, lanes)

        elif node_type == "Broadcast":
            value = parse_node(node_dict["value"])
            lanes = node_dict["lanes"]
            return BroadcastNode(value, lanes)

        else:
            raise ValueError(f"Unknown node type: {node_type}")

    # Parse the top-level list of function ASTs
    root_nodes = []
    for func_obj in ast_json_obj:
        function_name = func_obj["name"]
        ast_dict = func_obj["ast"]
        child_node = parse_node(ast_dict)
        root_nodes.append(RootNode(function_name, child_node))

    return root_nodes
