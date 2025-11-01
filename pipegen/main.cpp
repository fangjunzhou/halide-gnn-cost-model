#include <Halide.h>
#include <spdlog/spdlog.h>

#include "pipegen.h"

int main(int argc, char *argv[]) {
  // Get LOG_LEVEL from environment variable, default to "info".
  const char *logLevelEnv = std::getenv("LOG_LEVEL");
  spdlog::level::level_enum logLevel = spdlog::level::info;
  if (logLevelEnv) {
    std::string levelStr(logLevelEnv);
    std::transform(levelStr.begin(), levelStr.end(), levelStr.begin(),
                   ::tolower);
    if (levelStr == "trace") {
      logLevel = spdlog::level::trace;
    } else if (levelStr == "debug") {
      logLevel = spdlog::level::debug;
    } else if (levelStr == "info") {
      logLevel = spdlog::level::info;
    } else if (levelStr == "warn") {
      logLevel = spdlog::level::warn;
    } else if (levelStr == "error") {
      logLevel = spdlog::level::err;
    } else if (levelStr == "critical") {
      logLevel = spdlog::level::critical;
    } else if (levelStr == "off") {
      logLevel = spdlog::level::off;
    }
  }
  spdlog::set_level(logLevel);

  spdlog::info("Generating Halide pipeline");

  PipegenState state{.rng = std::mt19937(42)};
  Pipeline p;
  try {
    p = generatePipeline({.numArgs = 2, .maxFuncs = 16}, state);
  } catch (Halide::CompileError &e) {
    spdlog::error("Failed to generate pipeline: {}", e.what());
    return -1;
  }

  spdlog::info("Pipeline dag:");

  for (const auto &entry : p.parents) {
    spdlog::info("Function {} depend on:", entry.first);
    for (const auto &dep : entry.second) {
      spdlog::info("  - {}", dep.name());
    }
  }

  // Schedule the pipeline.
  try {
    schedulePipeline({}, p, state);
  } catch (Halide::CompileError &e) {
    spdlog::error("Failed to schedule pipeline: {}", e.what());
    return -1;
  }

  try {
    spdlog::info("Pipeline loop nest:");
    p.output.print_loop_nest();
  } catch (Halide::CompileError &e) {
    spdlog::error("Failed to compile loop nest: {}", e.what());
    return -1;
  }

  spdlog::info("Compiling the pipeline.");
  std::vector<Halide::Target> targets = {Halide::get_host_target()};
  p.halidePipeline.compile_to_multitarget_static_library(
      "pipelines/example/pipeline", {}, targets);
  p.halidePipeline.compile_to_lowered_stmt("pipelines/example/lowered.html", {},
                                           Halide::HTML);

  return 0;
}
