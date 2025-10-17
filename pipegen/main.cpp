#include <Halide.h>
#include <spdlog/spdlog.h>

#include <vector>

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

  spdlog::info("Compiling the pipeline.");
  std::vector<Halide::Target> targets = {Halide::get_host_target()};
  p.halidePipeline.compile_to_multitarget_static_library(
      "pipelines/example/pipeline", {}, targets);
  p.halidePipeline.compile_to_lowered_stmt("pipelines/example/lowered.html", {},
                                           Halide::HTML);

  return 0;
}
