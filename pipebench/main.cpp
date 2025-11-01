#include <Halide.h>
#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include "pipeline.h"

// Helper macros to concatenate `output` with the numeric pipeline index.
// Two-step expansion is required so `PIPELINE_IDX` (a macro) expands before
// token pasting with `##`.
#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define OUTPUT_FN(idx) CONCAT(output, idx)

static void pipelineBenchmark(benchmark::State &state) {
  Halide::Runtime::Buffer<int> out(state.range(0), state.range(0));
  for (auto _ : state) {
    OUTPUT_FN(PIPELINE_IDX)(out);
  }
}

BENCHMARK(pipelineBenchmark)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 12)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
