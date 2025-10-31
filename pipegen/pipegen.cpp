#include <Halide.h>
#include <spdlog/spdlog.h>

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

Pipeline generatePipeline(const PipegenConfig &config, PipegenState &state) {
  // Pipeline variables.
  std::vector<Halide::Var> vars;
  for (int i = 0; i < config.numArgs; i++) {
    vars.push_back(Halide::Var("v" + std::to_string(i)));
  }

  // Function pool.
  std::vector<Halide::Func> funcs;
  // Create the initial function.
  Halide::Func f("f0");
  f(vars) = generateExpr(vars, funcs, 1, state.rng);
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
    fi(vars) = generateExpr(vars, funcs, 1, state.rng);
    funcs.push_back(fi);
  }
  // The output function is the last function created.
  Pipeline p(funcs.back());
  return p;
}

std::vector<Halide::Var> splitArgs(Halide::Func &func, float splitProb,
                                   std::mt19937 &rng) {
  spdlog::debug("Splitting args for function {}", func.name());
  std::vector<Halide::Var> newArgs;
  for (auto arg : func.args()) {
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    if (probDist(rng) < splitProb) {
      // Random split factor in {2, 4, 8, 16}.
      std::vector<int> splitFactors = {2, 4, 8, 16};
      std::uniform_int_distribution<int> factorDist(0, splitFactors.size() - 1);
      int factor = splitFactors[factorDist(rng)];
      Halide::Var outer(arg.name() + "_out"), inner(arg.name() + "_in");
      func.split(arg, outer, inner, factor);
      newArgs.push_back(outer);
      newArgs.push_back(inner);
      spdlog::debug("  - Split arg {} into {} and {} with factor {}",
                    arg.name(), outer.name(), inner.name(), factor);
    } else {
      newArgs.push_back(arg);
    }
  }
  return newArgs;
}

void schedulePipeline(const ScheduleConfig &config, Pipeline &pipeline,
                      PipegenState &state) {
  for (auto &f : pipeline.funcs) {
    splitArgs(f, config.splitChance, state.rng);
  }
}
