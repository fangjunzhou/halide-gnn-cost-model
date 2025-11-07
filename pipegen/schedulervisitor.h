#pragma once
#include <Halide.h>

#include <nlohmann/json.hpp>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

class ScheduleJSONVisitor : public Halide::Internal::IRVisitor {
 public:
  ScheduleJSONVisitor(
      const std::map<std::string, Halide::Internal::Function> &e)
      : env(e) {};

  // The final nested schedule: an array of nodes at the root (like
  // PrintLoopNest lines)
  const nlohmann::json &result() const { return root_; }

 private:
  using Halide::Internal::IRVisitor::visit;

  // Root is an array of nodes
  nlohmann::json root_ = nlohmann::json::array();

  const std::map<std::string, Halide::Internal::Function> &env;
  Halide::Internal::Scope<Halide::Expr> constants;

  // Stack of current node objects (pending their "body" being filled)
  std::vector<nlohmann::json> node_stack_;
  // Parallel stack of "body arrays" we build for each node
  std::vector<nlohmann::json> children_stack_;  // each entry is a JSON array

  // Helpers --------------------------------------------------------------

  static std::string to_string(Halide::Internal::ForType t) {
    using Halide::Internal::ForType;
    switch (t) {
      case ForType::Serial:
        return "Serial";
      case ForType::Parallel:
        return "Parallel";
      case ForType::Unrolled:
        return "Unrolled";
      case ForType::Vectorized:
        return "Vectorized";
      case ForType::GPUBlock:
        return "GPUBlock";
      case ForType::GPUThread:
        return "GPUThread";
      case ForType::GPULane:
        return "GPULane";
      default:
        return "Unknown";
    }
  }

  static std::string to_string(Halide::DeviceAPI api) {
    using Halide::DeviceAPI;
    switch (api) {
      case DeviceAPI::None:
        return "None";
      case DeviceAPI::Host:
        return "Host";
      case DeviceAPI::OpenCL:
        return "OpenCL";
      case DeviceAPI::Metal:
        return "Metal";
      case DeviceAPI::CUDA:
        return "CUDA";
      case DeviceAPI::Hexagon:
        return "Hexagon";
      case DeviceAPI::D3D12Compute:
        return "D3D12Compute";
      case DeviceAPI::Vulkan:
        return "Vulkan";
      case DeviceAPI::WebGPU:
        return "WebGPU";
      default:
        return "Unknown";
    }
  }

  // Trim $n suffixes and stage/function prefixes, similar to PrintLoopNest
  static std::string simplify_name(const std::string &s, bool is_func) {
    std::ostringstream trimmed;
    bool keep = is_func;
    int dot_count = 0;
    for (size_t i = 0; i < s.size(); i++) {
      if (s[i] == '.') {
        dot_count++;
        if (dot_count >= 2) {
          if (dot_count == 2) {
            i++;
          }
          keep = true;
        }
      }
      if (s[i] == '$') {
        keep = false;
      }
      if (keep) {
        trimmed << s[i];
      }
    }
    return trimmed.str();
  }

  static std::string simplify_var_name(const std::string &s) {
    return simplify_name(s, /*is_func=*/false);
  }

  static std::string simplify_func_name(const std::string &s) {
    return simplify_name(s, /*is_func=*/true);
  }

  static std::string expr_to_string(const Halide::Expr &e) {
    std::ostringstream os;
    os << e;
    return os.str();
  }

  // Stack management -----------------------------------------------------

  void push_node(nlohmann::json node) {
    node_stack_.push_back(std::move(node));
    children_stack_.push_back(nlohmann::json::array());
  }

  void pop_node() {
    nlohmann::json children = std::move(children_stack_.back());
    children_stack_.pop_back();
    nlohmann::json node = std::move(node_stack_.back());
    node_stack_.pop_back();
    // attach children
    if (!children.is_null()) {
      node["body"] = std::move(children);
    }
    if (node_stack_.empty()) {
      root_.push_back(std::move(node));
    } else {
      children_stack_.back().push_back(std::move(node));
    }
  }

  void emit_leaf(nlohmann::json leaf) {
    if (node_stack_.empty()) {
      root_.push_back(std::move(leaf));
    } else {
      children_stack_.back().push_back(std::move(leaf));
    }
  }

  // Visitor overrides ----------------------------------------------------

  void visit(const Halide::Internal::For *op) override {
    nlohmann::json j;
    j["type"] = "For";
    j["for_type"] = to_string(op->for_type);
    j["var"] = simplify_var_name(op->name);
    // Bounds as strings (kept simple to avoid extra dependencies)
    Halide::Expr min_val = op->min, extent_val = op->extent;
    const Halide::Internal::Variable *min_var =
        min_val.as<Halide::Internal::Variable>();
    const Halide::Internal::Variable *extent_var =
        extent_val.as<Halide::Internal::Variable>();
    if (min_var) {
      if (const Halide::Expr *e = constants.find(min_var->name)) {
        min_val = *e;
      }
    }

    if (extent_var) {
      if (const Halide::Expr *e = constants.find(extent_var->name)) {
        extent_val = *e;
      }
    }

    if (extent_val.defined() && is_const(extent_val) && min_val.defined() &&
        is_const(min_val)) {
      Halide::Expr max_val = simplify(min_val + extent_val - 1);
      j["min"] = expr_to_string(min_val);
      j["max"] = expr_to_string(max_val);
    }
    push_node(std::move(j));
    op->body.accept(this);
    pop_node();
  }

  void visit(const Halide::Internal::Realize *op) override {
    // If the storage and compute levels for this function are
    // distinct, print the store level too.
    auto it = env.find(op->name);
    if (it != env.end() && !(it->second.schedule().store_level() ==
                             it->second.schedule().compute_level())) {
      nlohmann::json j;
      j["type"] = "Store";
      j["func"] = simplify_func_name(op->name);
      push_node(std::move(j));
      op->body.accept(this);
      pop_node();
    } else {
      op->body.accept(this);
    }
  }

  void visit(const Halide::Internal::ProducerConsumer *op) override {
    op->body.accept(this);
  }

  void visit(const Halide::Internal::Provide *op) override {
    nlohmann::json j;
    j["type"] = "Compute";
    j["func"] = simplify_func_name(op->name);
    emit_leaf(std::move(j));
  }

  void visit(const Halide::Internal::LetStmt *op) override {
    // Keep it simple: ignore Let bindings in the JSON, just traverse body.
    if (is_const(op->value)) {
      Halide::Internal::ScopedBinding<Halide::Expr> bind(constants, op->name,
                                                         op->value);
      op->body.accept(this);
    } else {
      op->body.accept(this);
    }
  }
};
