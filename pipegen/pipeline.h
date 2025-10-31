#pragma once

#include <Halide.h>

/**
 * @class Pipeline
 * @brief A custom wrapper over Halide::Pipeline for DAG management. The current
 * implementation only supports single output pipelines.
 *
 */
#include <string>
#include <unordered_map>
#include <vector>
class Pipeline {
 public:
  Pipeline(Halide::Func output);

  // Copy constructor
  Pipeline(const Pipeline &other);

  ~Pipeline();

  /**
   * @brief The output function of the pipeline.
   */
  Halide::Func output;

  /**
   * @brief A of functions in the pipeline.
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

 private:
};
