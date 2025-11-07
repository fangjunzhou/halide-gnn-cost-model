#pragma once

#include <Halide.h>

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @class Pipeline
 * @brief A custom wrapper over Halide::Pipeline for DAG management. The current
 * implementation only supports single output pipelines.
 *
 */
class Pipeline {
 public:
  Pipeline() = default;

  Pipeline(Halide::Func output);

  // Copy constructor
  Pipeline(const Pipeline &other);

  ~Pipeline();

  /**
   * @brief The output function of the pipeline.
   */
  Halide::Func output;

  /**
   * @brief A topologically sorted list of functions in the pipeline.
   */
  std::vector<Halide::Func> funcs;

  /**
   * @brief A mapping from function names to their parent(dependency) function
   * names.
   */
  std::unordered_map<std::string, std::vector<Halide::Func>> parents;

  /**
   * @brief A mapping from function names to their child(caller) function names.
   */
  std::unordered_map<std::string, std::vector<Halide::Func>> children;

  /**
   * @brief The underlying Halide pipeline.
   */
  Halide::Pipeline halidePipeline;

  /**
   * @brief Serialize the pipeline DAG into a json object.
   */
  nlohmann::json serializeDAG();

  /**
   * @brief Serialize the AST for each function into a json object.
   */
  nlohmann::json serializeAST();

  /**
   * @brief Serialize the scheduled loop-nest into a nested JSON object.
   * Returns an array of nodes at the root.
   */
  nlohmann::json serializeSchedule();

  // Store per-function JSON ASTs created at generation time.
  // Key: function name -> value: JSON AST for that function (or null)
  std::unordered_map<std::string, nlohmann::json> func_asts;

  // TODO: Add schedule serialization.

 private:
  void topologicalSort();
};