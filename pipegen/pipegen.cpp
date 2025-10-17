#include <Halide.h>

#include <random>
#include <vector>

#include "pipegen.h"

Halide::Expr generateExpr(const std::vector<Halide::Var> &vars,
                          const std::vector<Halide::Func> &funcs, int depth,
                          std::mt19937 &rng) {
  // Base case: return a variable, a constant, or a function call.
  if (depth == 0) {
    std::uniform_int_distribution<int> choiceDist(0,
                                                  vars.size() + funcs.size());
    int choice = choiceDist(rng);
    if (choice < vars.size()) {
      return vars[choice];
    } else if (choice < vars.size() + funcs.size()) {
      int funcIndex = choice - vars.size();
      Halide::Func f = funcs[funcIndex];
      return f(vars);
    } else {
      std::uniform_int_distribution<int> constDist(0, 10);
      return constDist(rng);
    }
  }
  // Recursive case: build a more complex expression.
  std::uniform_int_distribution<int> opDist(0, 3);
  int op = opDist(rng);
  Halide::Expr left = generateExpr(vars, funcs, depth - 1, rng);
  Halide::Expr right = generateExpr(vars, funcs, depth - 1, rng);
  switch (op) {
    case 0:
      return left + right;
    case 1:
      return left - right;
    case 2:
      return left * right;
    case 3:
      return left / (right + 1);  // Avoid division by zero.
    default:
      return left + right;
  }
}

Pipeline generatePipeline(const PipegenConfig &config) {
  std::mt19937 rng(config.seed);
  // Pipeline variables.
  std::vector<Halide::Var> vars;
  for (int i = 0; i < config.numArgs; i++) {
    vars.push_back(Halide::Var("v" + std::to_string(i)));
  }

  // Function pool.
  std::vector<Halide::Func> funcs;
  // Create the initial function.
  Halide::Func f("f0");
  f(vars) = generateExpr(vars, funcs, 1, rng);
  funcs.push_back(f);
  // Create the rest of the pipeline.
  for (int i = 1; i < config.maxFuncs; i++) {
    std::string funcName;
    if (i == config.maxFuncs - 1) {
      funcName = "output";
    } else {
      funcName = "f" + std::to_string(i);
    }
    Halide::Func fi(funcName);
    fi(vars) = generateExpr(vars, funcs, 1, rng);
    funcs.push_back(fi);
  }
  // The output function is the last function created.
  Pipeline p(funcs.back());
  return p;
}
