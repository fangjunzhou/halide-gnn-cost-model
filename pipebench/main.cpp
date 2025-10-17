#include <Halide.h>
#include <spdlog/spdlog.h>

#include "pipeline.h"

int main(int argc, char *argv[]) {
  spdlog::info("Starting Halide pipeline...");

  Halide::Runtime::Buffer<int> out(128, 128);
  h(out);

  spdlog::info("Pipeline completed successfully.");

  return 0;
}
