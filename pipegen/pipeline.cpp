#include <Halide.h>

#include "pipeline.h"

#include "astvisitor.h"

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
  // Return an array of functions with their name and the AST we collected during generation.
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
