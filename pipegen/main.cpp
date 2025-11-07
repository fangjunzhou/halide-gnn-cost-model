#include <argparse/argparse.hpp>
#include <Halide.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "pipegen.h"
#include "pipeline.h"

static spdlog::level::level_enum level_from_string(const std::string &s) {
  using spdlog::level::level_enum;
  std::string t;
  t.reserve(s.size());
  for (char c : s) t.push_back(static_cast<char>(::tolower(c)));
  if (t == "trace") return level_enum::trace;
  if (t == "debug") return level_enum::debug;
  if (t == "info") return level_enum::info;
  if (t == "warn" || t == "warning") return level_enum::warn;
  if (t == "error") return level_enum::err;
  if (t == "critical" || t == "fatal") return level_enum::critical;
  if (t == "off" || t == "none") return level_enum::off;
  return level_enum::info;
}

int main(int argc, char **argv) {
  argparse::ArgumentParser program("pipegen");

  program.add_argument("-n", "--num-schedules")
      .default_value(4)
      .scan<'i', int>()
      .help("number of pipelines to generate");

  program.add_argument("--log-level")
      .default_value(std::string("info"))
      .help("log level: trace|debug|info|warn|error|critical|off");

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    spdlog::error("{}", err.what());
    std::cout << program;
    return 1;
  }

  const std::string lvl = program.get<std::string>("--log-level");
  spdlog::set_level(level_from_string(lvl));
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
  spdlog::info("pipegen starting with log level: {}", lvl);

  const int numSchedules = program.get<int>("--num-schedules");
  spdlog::info("Will generate {} pipeline(s)", numSchedules);

  for (int i = 0; i < numSchedules; i++) {
    const std::string base_dir = "pipelines/pipeline_" + std::to_string(i);
    spdlog::info("Generating pipeline {} -> {}", i, base_dir);
    std::filesystem::create_directories(base_dir);

    try {
      PipegenConfig cfg;
      ScheduleConfig sc;
      PipegenState st;

      Pipeline p = generatePipeline(cfg, st);
      // NOTE: Fix call order to match signature: (const ScheduleConfig&, Pipeline&, PipegenState&)
      schedulePipeline(sc, p, st);

      // Compile artifacts
      std::string libBase = base_dir + "/pipeline";
      std::string stmtPath = base_dir + "/lowered.stmt";
      std::vector<Halide::Target> targets = {Halide::get_host_target()};
      p.halidePipeline.compile_to_multitarget_static_library(libBase, {}, targets);
      p.halidePipeline.compile_to_lowered_stmt(stmtPath, {}, Halide::Text);

      // DAG JSON
      {
        nlohmann::json dag = p.serializeDAG();
        std::ofstream out(base_dir + "/dag.json");
        out << dag.dump(4);
      }

      // AST JSON
      {
        nlohmann::json ast = p.serializeAST();
        std::ofstream out(base_dir + "/ast.json");
        out << ast.dump(4);
      }

      // Schedule JSON (nested)
      {
        nlohmann::json sched = p.serializeSchedule();
        std::ofstream out(base_dir + "/schedule.json");
        out << sched.dump(4);
      }

      spdlog::info("Pipeline {} done", i);

    } catch (const Halide::Error &e) {
      spdlog::error("Halide error: {}", e.what());
    } catch (const std::exception &e) {
      spdlog::error("Exception: {}", e.what());
    }
  }

  spdlog::info("All done");
  return 0;
}