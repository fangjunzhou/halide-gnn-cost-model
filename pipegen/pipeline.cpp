#include <Halide.h>

#include "pipeline.h"

Pipeline::Pipeline(Halide::Func output) {
  this->output = output;
  // DFS the output function to find all functions in the DAG.
  std::vector<Halide::Func> stack;
  stack.push_back(output);
  while (!stack.empty()) {
    Halide::Func f = stack.back();
    stack.pop_back();
    // Check if the function is already in the list.
    bool found = false;
    for (const auto &func : funcs) {
      if (func.name() == f.name()) {
        found = true;
        break;
      }
    }
    if (found) {
      continue;
    }
    funcs.push_back(f);
    // Get the dependencies of the function.
    auto dependencies = Halide::Internal::find_direct_calls(f.function());
    dag[f.name()] = {};
    for (const auto &dep : dependencies) {
      stack.push_back(Halide::Func(dep.second));
      dag[f.name()].push_back(dep.first);
    }
  }
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
