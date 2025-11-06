#pragma once

#include <Halide.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

// JSON AST visitor for Halide IR expressions.
//
// This visitor visits expression nodes and builds a nlohmann::json object
// representing the subtree for the visited node. It supports the main kinds
// used by the repo's generateExpr generator: Add, Sub, Mul, Div, Variable,
// IntImm, FloatImm, and Call. It uses an internal stack to accumulate child
// JSON nodes while traversing.
class JSONASTVisitor : public Halide::Internal::IRVisitor {
 public:
  JSONASTVisitor() = default;

  // Visit an expression and get the resulting JSON AST node.
  // Example:
  //   JSONASTVisitor v;
  //   nlohmann::json ast = v.ast_for(expr);
  nlohmann::json ast_for(const Halide::Expr &expr) {
    stack_.clear();
    expr->accept(this);
    if (stack_.empty()) {
      return nlohmann::json(nullptr);
    }
    nlohmann::json root = stack_.back();
    stack_.pop_back();
    return root;
  }

 protected:
  // Binary ops: visit children, then construct JSON node.
  void visit(const Halide::Internal::Add *op) override {
    op->a->accept(this);
    op->b->accept(this);
    // children are on stack: [..., left, right]
    nlohmann::json right = pop_stack();
    nlohmann::json left = pop_stack();
    nlohmann::json node;
    node["type"] = "Add";
    node["left"] = left;
    node["right"] = right;
    stack_.push_back(node);
  }

  void visit(const Halide::Internal::Sub *op) override {
    op->a->accept(this);
    op->b->accept(this);
    nlohmann::json right = pop_stack();
    nlohmann::json left = pop_stack();
    nlohmann::json node;
    node["type"] = "Sub";
    node["left"] = left;
    node["right"] = right;
    stack_.push_back(node);
  }

  void visit(const Halide::Internal::Mul *op) override {
    op->a->accept(this);
    op->b->accept(this);
    nlohmann::json right = pop_stack();
    nlohmann::json left = pop_stack();
    nlohmann::json node;
    node["type"] = "Mul";
    node["left"] = left;
    node["right"] = right;
    stack_.push_back(node);
  }

  void visit(const Halide::Internal::Div *op) override {
    op->a->accept(this);
    op->b->accept(this);
    nlohmann::json right = pop_stack();
    nlohmann::json left = pop_stack();
    nlohmann::json node;
    node["type"] = "Div";
    node["left"] = left;
    node["right"] = right;
    stack_.push_back(node);
  }

  // Variable
  void visit(const Halide::Internal::Variable *op) override {
    nlohmann::json node;
    node["type"] = "Variable";
    node["name"] = op->name;
    stack_.push_back(node);
  }

  // Integer constant
  void visit(const Halide::Internal::IntImm *op) override {
    nlohmann::json node;
    node["type"] = "IntImm";
    node["value"] = op->value;
    stack_.push_back(node);
  }

  // Float constant
  void visit(const Halide::Internal::FloatImm *op) override {
    nlohmann::json node;
    node["type"] = "FloatImm";
    node["value"] = op->value;
    stack_.push_back(node);
  }

  // Call (function call). Collect argument JSONs.
  void visit(const Halide::Internal::Call *op) override {
    // Visit all args in order.
    size_t start = stack_.size();
    for (const auto &arg : op->args) {
      arg->accept(this);
    }
    // args are appended to stack; copy them in order.
    std::vector<nlohmann::json> args_json;
    for (size_t i = start; i < stack_.size(); ++i) {
      args_json.push_back(stack_[i]);
    }
    // pop the arg entries from the stack.
    stack_.resize(start);

    nlohmann::json node;
    node["type"] = "Call";
    node["name"] = op->name;
    // call_type: Halide / Intrinsic / Extern / Image / PureExtern etc.
    switch (op->call_type) {
      case Halide::Internal::Call::Halide:
        node["call_type"] = "Halide";
        break;
      case Halide::Internal::Call::Intrinsic:
        node["call_type"] = "Intrinsic";
        break;
      case Halide::Internal::Call::Extern:
        node["call_type"] = "Extern";
        break;
      case Halide::Internal::Call::Image:
        node["call_type"] = "Image";
        break;
      case Halide::Internal::Call::PureExtern:
        node["call_type"] = "PureExtern";
        break;
      default:
        node["call_type"] = "Unknown";
        break;
    }
    node["args"] = nlohmann::json::array();
    for (auto &a : args_json) node["args"].push_back(a);

    stack_.push_back(node);
  }

  // Cast: visit the value then wrap it. Different Halide versions expose
  // different helpers; avoid calling to_string() on Type which may not exist.
  void visit(const Halide::Internal::Cast *op) override {
    op->value->accept(this);
    nlohmann::json val = pop_stack();
    nlohmann::json node;
    node["type"] = "Cast";
    // Record type by composing a small string with code/bits/lanes (portable).
    Halide::Type t = op->value.type();
    std::string tstr = "code:" + std::to_string(static_cast<int>(t.code())) +
                       ",bits:" + std::to_string(t.bits()) +
                       ",lanes:" + std::to_string(t.lanes());
    node["type_name"] = tstr;
    node["value"] = val;
    stack_.push_back(node);
  }

  void visit(const Halide::Internal::Ramp *op) override {
    op->base->accept(this);
    op->stride->accept(this);
    nlohmann::json stride = pop_stack();
    nlohmann::json base = pop_stack();
    nlohmann::json node;
    node["type"] = "Ramp";
    node["base"] = base;
    node["stride"] = stride;
    node["lanes"] = op->lanes;
    stack_.push_back(node);
  }

  void visit(const Halide::Internal::Broadcast *op) override {
    op->value->accept(this);
    nlohmann::json val = pop_stack();
    nlohmann::json node;
    node["type"] = "Broadcast";
    node["value"] = val;
    node["lanes"] = op->lanes;
    stack_.push_back(node);
  }

  // NOTE: we do not define visit(const Halide::Internal::Node*). Older/newer
  // Halide versions differ here; leaving that method out avoids referencing
  // a non-existent type. If you run into an unhandled node kind, we'll add
  // the specific handler for that node.

 private:
  // Helper to pop last json on stack with safety.
  nlohmann::json pop_stack() {
    if (stack_.empty()) {
      return nullptr;
    }
    nlohmann::json j = stack_.back();
    stack_.pop_back();
    return j;
  }

  std::vector<nlohmann::json> stack_;
};