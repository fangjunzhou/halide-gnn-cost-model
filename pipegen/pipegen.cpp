#include <Halide.h>
#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "astvisitor.h"
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
  // Map to collect per-function ASTs while generating
  std::unordered_map<std::string, nlohmann::json> ast_map;

  // Create the initial function.
  {
    Halide::Func f("f0");
    Halide::Expr expr = generateExpr(vars, funcs, 1, state.rng);
    f(vars) = expr;
    funcs.push_back(f);

    // Build JSON AST for this function and store.
    JSONASTVisitor jv;
    nlohmann::json ast = jv.ast_for(expr);
    ast_map[f.name()] = ast;
  }

  // Create the rest of the pipeline.
  for (int i = 1; i < config.maxFuncs; i++) {
    std::string funcName;
    if (i == config.maxFuncs - 1) {
      funcName = config.outputFuncName;
    } else {
      funcName = "f" + std::to_string(i);
    }
    Halide::Func fi(funcName);
    Halide::Expr expr = generateExpr(vars, funcs, 1, state.rng);
    fi(vars) = expr;
    funcs.push_back(fi);

    // Build JSON AST for this function and store.
    JSONASTVisitor jv;
    nlohmann::json ast = jv.ast_for(expr);
    ast_map[fi.name()] = ast;
  }
  // The output function is the last function created.

  Pipeline p(funcs.back());
  // Attach the collected ASTs to the Pipeline so serializeAST can simply return
  // them.
  p.func_asts = std::move(ast_map);
  return p;
}

static std::vector<Halide::Var> splitFunc(Halide::Func &func, float splitProb,
                                          std::mt19937 &rng) {
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
      spdlog::debug("Split arg {} into {} and {} with factor {}", arg.name(),
                    outer.name(), inner.name(), factor);
    } else {
      newArgs.push_back(arg);
    }
  }
  return newArgs;
}

// Shuffle/reorder args and apply the reorder to the given function.
// Returns the VarOrRVar vector used for the reorder call.
static std::vector<Halide::VarOrRVar> reorderAndShuffleArgs(
    Halide::Func &f, std::vector<Halide::Var> &args, std::mt19937 &rng) {
  std::shuffle(args.begin(), args.end(), rng);
  std::vector<Halide::VarOrRVar> varArgs;
  std::string argNames;
  for (auto &arg : args) {
    varArgs.push_back(arg);
    argNames += arg.name() + " ";
  }
  f.reorder(varArgs);
  spdlog::debug("Reordered function {} with args {}", f.name(), argNames);
  return varArgs;
}

// Possibly vectorize the innermost loop based on naming convention and chance.
static void maybeVectorizeInnermost(Halide::Func &f,
                                    const std::vector<Halide::Var> &args,
                                    const ScheduleConfig &config,
                                    std::mt19937 &rng) {
  if (args.size() == 0) {
    return;
  }
  auto innermostArg = args[0];
  std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
  if (innermostArg.name().find("_in") != std::string::npos &&
      probDist(rng) < config.vectorizeInnermostChance) {
    // Preserve original behavior: vectorize the last arg.
    f.vectorize(innermostArg);
    spdlog::debug("Vectorized function {} on arg {}", f.name(),
                  innermostArg.name());
  }
}

// Possibly parallelize the outermost loop based on chance.
static bool maybeParallelizeOutermost(Halide::Func &f,
                                      const std::vector<Halide::Var> &args,
                                      const ScheduleConfig &config,
                                      std::mt19937 &rng) {
  if (args.size() == 0) {
    return false;
  }
  std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
  if (probDist(rng) < config.parallelizeOutermostChance) {
    // Preserve original behavior: parallelize the last arg.
    f.parallel(args.back());
    spdlog::debug("Parallelized function {} on arg {}", f.name(),
                  args.back().name());
    return true;
  }
  return false;
}

// Cross-stage scheduling: decide whether to compute at root or at a child
// loop level and whether to store at a particular child loop level.
// Returns true when the function was scheduled compute_root.
static std::optional<std::string> scheduleCrossStage(
    const ScheduleConfig &config, Pipeline &pipeline, Halide::Func &f,
    std::unordered_map<std::string, std::vector<Halide::Var>> &loopArgMap,
    std::unordered_map<std::string, bool> &parallelizedMap,
    PipegenState &state) {
  auto &children = pipeline.children[f.name()];
  // Schedule at root.
  std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
  if (probDist(state.rng) < config.computeAtRootChance ||
      children.size() != 1) {
    f.compute_root();
    spdlog::debug("Scheduled function {} compute at root", f.name());
    return std::nullopt;
  } else {
    auto &child = children[0];
    auto &loopArgs = loopArgMap[child.name()];
    std::string loopArgNames;
    for (auto &arg : loopArgs) {
      loopArgNames += arg.name() + " ";
    }
    spdlog::debug("Function {} has child {} with loop args {}", f.name(),
                  child.name(), loopArgNames);
    int maxLevel;
    if (parallelizedMap[child.name()]) {
      // Store below the outermost parallel loop.
      maxLevel = loopArgs.size() - 2;
    } else {
      maxLevel = loopArgs.size();
    }
    // Randomly pick a compute level.
    std::uniform_int_distribution<int> levelDist(0, maxLevel);
    int level = levelDist(state.rng);
    if (level >= loopArgs.size()) {
      f.compute_root();
      spdlog::debug("Scheduled function {} compute at root", f.name());
      return std::nullopt;
    }
    f.compute_at(child, loopArgs[level]);
    spdlog::debug("Scheduled function {} compute at {} - {}", f.name(),
                  child.name(), loopArgs[level].name());
    std::uniform_real_distribution<float> storeProbDist(0.0f, 1.0f);
    // Randomly pick a storage level.
    std::uniform_int_distribution<int> storeLevelDist(level, maxLevel);
    int storeLevel = storeLevelDist(state.rng);
    if (storeLevel >= loopArgs.size()) {
      f.store_root();
      spdlog::debug("Scheduled function {} store at root", f.name());
    } else {
      f.store_at(child, loopArgs[storeLevel]);
      spdlog::debug("Scheduled function {} store at {} - {}", f.name(),
                    child.name(), loopArgs[storeLevel].name());
    }
    return child.name();
  }
}

void schedulePipeline(const ScheduleConfig &config, Pipeline &pipeline,
                      PipegenState &state) {
  std::unordered_map<std::string, std::vector<Halide::Var>> loopArgMap;
  std::unordered_map<std::string, bool> parallelizedMap;
  for (auto &f : pipeline.funcs) {
    spdlog::debug("- Scheduling function {}", f.name());

    /* ----------------- Cross-Stage Scheduling ----------------- */

    auto parent = scheduleCrossStage(config, pipeline, f, loopArgMap,
                                     parallelizedMap, state);

    /* ----------------- Intra-Stage Scheduling ----------------- */

    // Randomly split args.
    auto args = splitFunc(f, config.splitChance, state.rng);
    // Shuffle/reorder and apply reorder call.
    auto varArgs = reorderAndShuffleArgs(f, args, state.rng);
    // Randomly vectorize innermost loop.
    maybeVectorizeInnermost(f, args, config, state.rng);
    // Randomly parallelize outermost loop.
    if (!parent) {
      bool parallelized = maybeParallelizeOutermost(f, args, config, state.rng);
      parallelizedMap[f.name()] = parallelized;
    } else {
      parallelizedMap[f.name()] = parallelizedMap[*parent];
    }

    loopArgMap[f.name()] = args;
  }
}
