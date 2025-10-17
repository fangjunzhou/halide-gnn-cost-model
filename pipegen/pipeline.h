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
   * @brief A DAG representation of the pipeline.
   */
  std::unordered_map<std::string, std::vector<std::string>> dag;

  /**
   * @brief The underlying Halide pipeline.
   */
  Halide::Pipeline halidePipeline;

 private:
};
