#pragma once

#include <Halide.h>

#include "pipeline.h"

struct PipegenState {
  std::mt19937 rng;
};

struct PipegenConfig {
  // Number of arguments in the pipeline
  int numArgs = 2;
  // Maximum number of functions in the pipeline
  int maxFuncs = 8;
};

struct ScheduleConfig {
  // Probity of splitting an argument.
  float splitChance = 0.3f;
};

/**
 * @brief Generate random pipeline.
 *
 * @param config pipeline generation config.
 * @param state pipeline generation state.
 * @return generated pipeline.
 */
Pipeline generatePipeline(const PipegenConfig &config, PipegenState &state);

/**
 * @brief Randomly schedule the pipeline.
 *
 * @param pipeline The pipeline to schedule.
 * @param state The pipegen state.
 */
void schedulePipeline(const ScheduleConfig &config, Pipeline &pipeline,
                      PipegenState &state);
