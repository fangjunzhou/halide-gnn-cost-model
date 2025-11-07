#include <Halide.h>

#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "pipeline.h"
#include "schedulervisitor.h"

Pipeline::Pipeline(Halide::Func output) {
  this->output = output;
  // DFS the output function to find all functions in the DAG.
  std::vector<Halide::Func> stack;
  std::vector<Halide::Func> visited;
  stack.push_back(output);
  while (!stack.empty()) {
    Halide::Func f = stack.back();
    stack.pop_back();
    // Check if the function is already in the list.
    bool found = false;
    for (const auto &func : visited) {
      if (func.name() == f.name()) {
        found = true;
        break;
      }
    }
    if (found) {
      continue;
    }
    visited.push_back(f);
    // Get the dependencies of the function.
    auto dependencies = Halide::Internal::find_direct_calls(f.function());
    parents[f.name()] = {};
    for (const auto &dep : dependencies) {
      Halide::Func depFunc(dep.second);
      stack.push_back(depFunc);
      parents[f.name()].push_back(depFunc);
      if (children.find(dep.first) == children.end()) {
        children[dep.first] = {};
      }
      children[dep.first].push_back(f);
    }
  }
  // Topologically sort the functions.
  topologicalSort();
  halidePipeline = Halide::Pipeline(output);
}

Pipeline::Pipeline(const Pipeline &other) {
  // Deep copy the output function.
  Halide::Func copiedOutput("output");
  std::map<Halide::Internal::FunctionPtr, Halide::Internal::FunctionPtr> copies;
  other.output.function().deep_copy(copiedOutput.function().get_contents(),
                                    copies);
  *this = Pipeline(copiedOutput);
}

Pipeline::~Pipeline() {}

nlohmann::json Pipeline::serializeDAG() {
  nlohmann::json j = nlohmann::json::array();
  for (auto &func : this->funcs) {
    nlohmann::json funcJson;
    funcJson["name"] = func.name();
    funcJson["parents"] = nlohmann::json::array();
    for (auto &parent : this->parents[func.name()]) {
      funcJson["parents"].push_back(parent.name());
    }
    j.push_back(funcJson);
  }
  return j;
}

nlohmann::json Pipeline::serializeAST() {
  // Return an array of functions with their name and the AST we collected
  // during generation.
  nlohmann::json root = nlohmann::json::array();
  for (auto &func : this->funcs) {
    nlohmann::json f;
    f["name"] = func.name();
    auto it = func_asts.find(func.name());
    if (it != func_asts.end()) {
      f["ast"] = it->second;
    } else {
      f["ast"] = nullptr;
    }
    root.push_back(f);
  }
  return root;
}

nlohmann::json Pipeline::serializeSchedule() {
  // Do the first part of lowering:
  std::vector<Halide::Internal::Function> output_funcs{this->output.function()};

  // Create a deep-copy of the entire graph of Funcs.
  auto [outputs, env] = Halide::Internal::deep_copy(
      output_funcs, Halide::Internal::build_environment(output_funcs));

  // Output functions should all be computed and stored at root.
  for (const Halide::Internal::Function &f : outputs) {
    Halide::Func(f).compute_root().store_root();
  }

  // Finalize all the LoopLevels
  for (auto &iter : env) {
    iter.second.lock_loop_levels();
  }

  // Substitute in wrapper Funcs
  env = Halide::Internal::wrap_func_calls(env);

  // Compute a realization order and determine group of functions which loops
  // are to be fused together
  auto [order, fused_groups] =
      Halide::Internal::realization_order(outputs, env);

  // Try to simplify the RHS/LHS of a function definition by propagating its
  // specializations' conditions
  Halide::Internal::simplify_specializations(env);

  // For the purposes of printing the loop nest, we don't want to
  // worry about which features are and aren't enabled.
  Halide::Target target = Halide::get_host_target();
  for (Halide::DeviceAPI api : Halide::all_device_apis) {
    target.set_feature(
        Halide::target_feature_for_device_api(Halide::DeviceAPI(api)));
  }

  bool any_memoized = false;
  // Schedule the functions.
  Halide::Internal::Stmt s = Halide::Internal::schedule_functions(
      outputs, fused_groups, env, target, any_memoized);

  // Compute the maximum and minimum possible value of each
  // function. Used in later bounds inference passes.
  Halide::Internal::FuncValueBounds func_bounds =
      Halide::Internal::compute_function_value_bounds(order, env);

  // This pass injects nested definitions of variable names, so we
  // can't simplify statements from here until we fix them up. (We
  // can still simplify Exprs).
  s = Halide::Internal::bounds_inference(s, outputs, order, fused_groups, env,
                                         func_bounds, target);
  s = Halide::Internal::remove_extern_loops(s);
  s = Halide::Internal::sliding_window(s, env);
  s = Halide::Internal::simplify_correlated_differences(s);
  s = Halide::Internal::allocation_bounds_inference(s, env, func_bounds);
  s = Halide::Internal::remove_undef(s);
  s = Halide::Internal::uniquify_variable_names(s);
  s = Halide::Internal::simplify(s, false);

  // Use public API only: print the loop nest and parse to nested JSON.
  ScheduleJSONVisitor visitor(env);
  s.accept(&visitor);
  return visitor.result();
}

void Pipeline::topologicalSort() {
  std::unordered_map<std::string, int> inDegree;
  for (const auto &entry : children) {
    inDegree[entry.first] = entry.second.size();
  }
  std::vector<Halide::Func> orphans = {this->output};
  funcs.clear();

  while (!orphans.empty()) {
    Halide::Func f = orphans.back();
    orphans.pop_back();
    funcs.push_back(f);
    for (const auto &parent : parents[f.name()]) {
      inDegree[parent.name()]--;
      if (inDegree[parent.name()] == 0) {
        orphans.push_back(parent);
      }
    }
  }
}
