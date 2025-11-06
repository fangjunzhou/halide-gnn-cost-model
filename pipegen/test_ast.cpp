#include <Halide.h>
#include <iostream>
// #include "nlohmann/json.hpp"

// Include the JSON AST visitor
#include "astvisitor.h"

int main() {
    Halide::Var v0("v0"), v1("v1");
    Halide::Expr expr = v0 + v1;

    JSONASTVisitor visitor;
    nlohmann::json ast = visitor.ast_for(expr);
    std::cout << ast.dump(2) << std::endl;
    return 0;
}