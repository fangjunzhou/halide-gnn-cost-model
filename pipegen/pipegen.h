#pragma once

#include <Halide.h>

#include "pipeline.h"

struct PipegenConfig {
  // Random seed
  int seed = 42;
  // Number of arguments in the pipeline
  int numArgs = 2;
  // Maximum number of functions in the pipeline
  int maxFuncs = 8;
};

/**
 * @brief Generates a random Halide pipeline.
 */
Pipeline generatePipeline(const PipegenConfig &config = PipegenConfig());
