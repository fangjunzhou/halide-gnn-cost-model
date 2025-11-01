#include <Halide.h>
#include <spdlog/spdlog.h>

#include <vector>
#include <filesystem>
#include <exception>
#include <iostream>

#include "pipegen.h"

int main(int argc, char *argv[]) {
  spdlog::info("Generating Halide pipeline");

  Pipeline p = generatePipeline({.numArgs = 2, .maxFuncs = 16});

  spdlog::info("Pipeline dag:");

  for (const auto &entry : p.dag) {
    spdlog::info("Function {} calls:", entry.first);
    for (const auto &dep : entry.second) {
      spdlog::info("  - {}", dep);
    }
  }

  for (auto &func : p.funcs) {
    spdlog::info("Scheduling {} at root", func.name());
    func.compute_root();
  }

  spdlog::info("Pipeline loop nest:");
  p.output.print_loop_nest();

  // IMPORTANT: do not invoke Halide code generation here. We only want the lowered IR.
  spdlog::info("Emitting lowered IR (text and HTML). Skipping code generation.");

  // Ensure output directory exists
  std::filesystem::create_directories("pipelines/example");

  try {
    // Emit HTML lowered IR (visual)
    p.halidePipeline.compile_to_lowered_stmt("pipelines/example/lowered.html", {},
                                             Halide::HTML);

    // Emit textual lowered IR (the AST text you requested)
    p.halidePipeline.compile_to_lowered_stmt("pipelines/example/lowered.txt", {},
                                             Halide::Text);

    spdlog::info("Wrote pipelines/example/lowered.txt and lowered.html");
  } catch (const std::exception &e) {
    spdlog::error("Exception while emitting lowered IR: {}", e.what());
    std::cerr << "Exception while emitting lowered IR: " << e.what() << std::endl;
    return 2;
  } catch (...) {
    spdlog::error("Unknown exception while emitting lowered IR");
    std::cerr << "Unknown exception while emitting lowered IR" << std::endl;
    return 3;
  }

  return 0;
}
