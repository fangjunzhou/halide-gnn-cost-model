#include <Halide.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "astvisitor.h"

// Small local generator that mirrors pipegen::generateExpr behaviour
// so we can produce complex random Expr trees for testing the JSON visitor.
//
// This generator supports:
//  - choosing between variables, constants, or function calls (if funcs present)
//  - binary ops: +, -, *, / (with denominator +1 to avoid div-by-zero)
//
// We create a tiny pool of functions (Halide::Func) so Call nodes may appear.

using namespace Halide;

Expr generate_random_expr(const std::vector<Var> &vars,
                          const std::vector<Func> &funcs, int depth,
                          std::mt19937 &rng) {
  if (depth == 0) {
    std::uniform_int_distribution<int> choiceDist(0, (int)vars.size() + (int)funcs.size());
    int choice = choiceDist(rng);
    if (choice < (int)vars.size()) {
      return vars[choice];
    } else if (choice < (int)vars.size() + (int)funcs.size()) {
      int funcIndex = choice - (int)vars.size();
      Func f = funcs[funcIndex];
      return f(vars);
    } else {
      std::uniform_int_distribution<int> constDist(0, 10);
      return constDist(rng);
    }
  }
  std::uniform_int_distribution<int> opDist(0, 3);
  int op = opDist(rng);
  Expr left = generate_random_expr(vars, funcs, depth - 1, rng);
  Expr right = generate_random_expr(vars, funcs, depth - 1, rng);
  switch (op) {
    case 0:
      return left + right;
    case 1:
      return left - right;
    case 2:
      return left * right;
    case 3:
      return left / (right + 1);
    default:
      return left + right;
  }
}

int main() {
  // Random seed
  std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());

  // Build variables
  std::vector<Var> vars;
  vars.push_back(Var("v0"));
  vars.push_back(Var("v1"));
  vars.push_back(Var("v2"));

  // Create a small pool of Halide funcs that can be called (simple definitions).
  std::vector<Func> funcs;
  {
    Func f0("f0");
    // f0(v0,v1,v2) = v0 + 3
    std::vector<Var> argvars = vars;
    f0(argvars) = vars[0] + 3;
    funcs.push_back(f0);
  }
  {
    Func f1("f1");
    std::vector<Var> argvars = vars;
    f1(argvars) = vars[1] * 2;
    funcs.push_back(f1);
  }

  // Generate a complex expression
  Expr e = generate_random_expr(vars, funcs, /*depth=*/4, rng);

  // Produce JSON AST
  JSONASTVisitor visitor;
  nlohmann::json ast = visitor.ast_for(e);

  // Pretty print to stdout
  std::cout << ast.dump(2) << std::endl;

  // Also write to build/pipelines/random_ast.json (create dir if needed)
  std::string outdir = "pipelines/test_generated";
  std::filesystem::create_directories(outdir);
  std::ofstream ofs(outdir + "/random_ast.json");
  ofs << ast.dump(2) << std::endl;
  ofs.close();

  std::cout << "Wrote AST to " << outdir << "/random_ast.json" << std::endl;
  return 0;
}