#include <Halide.h>
#include <spdlog/spdlog.h>
#include <stdio.h>

#include <argparse/argparse.hpp>
#include <filesystem>

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

  // Parse command-line arguments. Allow the user to control how many
  // schedules/pipelines to generate via -n / --num-schedules (default: 4).
  argparse::ArgumentParser program("pipegen");
  program.add_argument("-n", "--num-schedules")
      .default_value(4)
      .scan<'i', int>()
      .help("number of schedules to generate");
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &e) {
    spdlog::error("Argument parse error: {}", e.what());
    return -1;
  }
  int numSchedules = program.get<int>("--num-schedules");
  if (numSchedules < 1) {
    spdlog::error("--num-schedules must be >= 1");
    return -1;
  }

  PipegenState state{.rng = std::mt19937(42)};

  for (int i = 0; i < numSchedules; i++) {
    spdlog::info("Generating pipeline {}", i);
    std::string outputFuncName = "output" + std::to_string(i);
    Pipeline p;
    try {
      p = generatePipeline(
          {.numArgs = 2, .maxFuncs = 16, .outputFuncName = outputFuncName},
          state);
    } catch (Halide::CompileError &e) {
      spdlog::error("Failed to generate pipeline: {}", e.what());
      return -1;
    }

    // Schedule the pipeline.
    spdlog::info("Scheduling the pipeline.");
    try {
      schedulePipeline({}, p, state);
    } catch (Halide::Error &e) {
      spdlog::error("Failed to schedule pipeline: {}", e.what());
      return -1;
    }

    spdlog::info("Compiling the pipeline.");
    std::string libName =
        "pipelines/pipeline_" + std::to_string(i) + "/pipeline";
    std::string stmtName =
        "pipelines/pipeline_" + std::to_string(i) + "/lowered.stmt";
    // Make sure the output directory exists.
    std::filesystem::create_directories("pipelines/pipeline_" +
                                        std::to_string(i));
    try {
      std::vector<Halide::Target> targets = {Halide::get_host_target()};
      p.halidePipeline.compile_to_multitarget_static_library(libName, {},
                                                             targets);
      p.halidePipeline.compile_to_lowered_stmt(stmtName, {}, Halide::Text);
    } catch (Halide::Error &e) {
      spdlog::error("Failed to compile loop nest: {}", e.what());
      return -1;
    }
  }

  return 0;
}
