#pragma once

#include <Halide.h>

#include <iostream>

class ASTVisitor : public Halide::Internal::IRVisitor {
 protected:
  void visit(const Halide::Internal::Add *op) override {
    std::cout << "add(";
    op->a->accept(this);
    std::cout << ",";
    op->b->accept(this);
    std::cout << ")";
  }

  void visit(const Halide::Internal::Variable *op) override {
    std::cout << op->name;
  }

  // TODO:Implement other visitor.
};
