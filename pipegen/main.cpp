#include <Halide.h>
#include <spdlog/spdlog.h>
#include <stdio.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>

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
  // DAG random seed.
  program.add_argument("--dag-seed")
      .default_value(42)
      .scan<'i', int>()
      .help("random seed for DAG generation");
  // Schedule random seed.
  program.add_argument("--schedule-seed")
      .default_value(42)
      .scan<'i', int>()
      .help("random seed for scheduling");
  // Output directory.
  program.add_argument("-o", "--output-dir")
      .default_value(std::string("pipelines/default"))
      .help("output directory for the generated pipeline");
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &e) {
    spdlog::error("Argument parse error: {}", e.what());
    return -1;
  }

  int dagSeed = program.get<int>("--dag-seed");
  int scheduleSeed = program.get<int>("--schedule-seed");
  std::string outputDir = program.get<std::string>("--output-dir");
  // Make output directory if it doesn't exist.
  std::filesystem::create_directories(outputDir);

  PipegenState state;

  spdlog::info("Generating pipeline");
  std::string outputFuncName = "output";
  Pipeline p;
  try {
    state.rng = std::mt19937(dagSeed);
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
    state.rng = std::mt19937(scheduleSeed);
    schedulePipeline({}, p, state);
  } catch (Halide::Error &e) {
    spdlog::error("Failed to schedule pipeline: {}", e.what());
    return -1;
  }

  spdlog::info("Compiling the pipeline.");
  std::string libName = outputDir + "/pipeline";
  std::string stmtName = outputDir + "/pipeline.stmt";
  try {
    std::vector<Halide::Target> targets = {Halide::get_host_target()};
    p.halidePipeline.compile_to_multitarget_static_library(libName, {},
                                                           targets);
    p.halidePipeline.compile_to_lowered_stmt(stmtName, {}, Halide::Text);
  } catch (Halide::Error &e) {
    spdlog::error("Failed to compile loop nest: {}", e.what());
    return -1;
  }
  // Save function dag json here.
  auto dagJson = p.serializeDAG();
  std::string dagJsonPath = outputDir + "/dag.json";
  std::ofstream dagJsonFile(dagJsonPath);
  dagJsonFile << dagJson.dump(4);
  dagJsonFile.close();
  // Save pipeline functions to AST json here.
  auto astJson = p.serializeAST();
  std::string astJsonPath = outputDir + "/ast.json";
  std::ofstream astJsonFile(astJsonPath);
  astJsonFile << astJson.dump(4);
  astJsonFile.close();
  // TODO: Save schedule json here.
  auto scheduleJson = p.serializeSchedule();
  std::string scheduleJsonPath = outputDir + "/schedule.json";
  std::ofstream scheduleJsonFile(scheduleJsonPath);
  scheduleJsonFile << scheduleJson.dump(4);
  scheduleJsonFile.close();

  return 0;
}
