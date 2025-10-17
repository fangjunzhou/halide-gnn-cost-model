#include <Halide.h>
#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include "pipeline.h"

static void pipelineBenchmark(benchmark::State &state) {
  Halide::Runtime::Buffer<int> out(state.range(0), state.range(0));
  for (auto _ : state) {
    output(out);
  }
}

BENCHMARK(pipelineBenchmark)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 12)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
