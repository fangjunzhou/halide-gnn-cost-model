# Halide GNN Cost Model

## Halide Pipeline Generation

This project contains tools to (1) generate Halide pipelines with randomized DAGs and schedules (`pipegen`) and (2) run micro-benchmarks against compiled pipelines (`pipebench`). The typical workflow is:

- Build `pipegen` and run it to emit a pipeline (static library, lowered statement, and JSON artifacts) into an output directory.
- (Optional) Build or link a small benchmark harness that calls the pipeline's `output(...)` function, or use the provided pipeline-specific benchmark binaries in `build/bin/`.

The generated pipeline files are written to the output directory you pass to `pipegen` (default: `pipelines/default`) and include:

- `pipeline` (the compiled static library produced by Halide)
- `pipeline.stmt` (lowered loop nest as Text)
- `dag.json` (function DAG)
- `ast.json` (AST for pipeline functions)
- `schedule.json` (schedule metadata)

### Dependencies

Required build/runtime dependencies:

- CMake (>= 3.29)
- A C++17 toolchain (compiler, make/ninja)
- Halide (host-target Halide dev package)
- spdlog (logging)
- google-benchmark (benchmarking library)
- nlohmann_json (JSON for Modern C++)

On macOS with Homebrew you can install many dependencies with (examples):

```bash
brew install cmake spdlog nlohmann-json google-benchmark
# Halide may be installed from a package or built from source; check
# https://halide-lang.org for platform-specific installation instructions
```

If you use Nix (recommended for reproducible dev environments), this repository provides a `flake.nix` dev shell that already contains the tools listed above. Enter the development environment with:

```bash
# Enter the flake dev shell (requires a recent nix with flakes enabled)
nix develop

# Or explicitly request the default devShell in the repo
nix develop .#default
```

The `flake.nix` defines a `devShell` with CMake, Halide, spdlog, argparse, benchmark, nlohmann_json and Python; once inside the dev shell you can run `cmake`/`cmake --build` as shown below.

### Generate pipelines (build & run)

This repository uses a CMake-driven workflow. The CMake top-level defines a `BUILD_TARGET` cache variable which selects which executable to build: `pipegen` or `pipebench`.

1) Configure & build `pipegen` (pipeline generator)

```bash
# Configure the build for pipegen
cmake -S . -B build -DBUILD_TARGET=pipegen

# Build the selected target
cmake --build build -j

# Binary will be in build/bin/pipegen
./build/bin/pipegen --help
```

Run `pipegen` to generate a pipeline. Useful flags (from `pipegen/main.cpp`):

- `--dag-seed <int>` — RNG seed for DAG generation (default: 42)
- `--schedule-seed <int>` — RNG seed for scheduling (default: 42)
- `-o, --output-dir <path>` — output directory for the generated pipeline
	(default: `pipelines/default`)

Example:

```bash
# Generate one pipeline into pipelines/pipeline_1_2
./build/bin/pipegen -o pipelines/pipeline_1_2 --dag-seed 1 --schedule-seed 2

# Inspect outputs
ls -la pipelines/pipeline_1_2
# -> pipeline (static lib), pipeline.stmt, dag.json, ast.json, schedule.json
```

2) Build & run benchmarks (`pipebench`)

You can build the benchmark harness with CMake using the `pipebench` target:

```bash
# Configure for pipebench
cmake -S . -B build -DBUILD_TARGET=pipebench -DPIPES_DIR=<pipelines_dir>
cmake --build build -j

# Run the benchmark driver (if it links against a pipeline)
./build/bin/pipebench_<pipeline_name>
```

Notes on benchmarking generated pipelines:

- When generating pipelines with `pipegen`, you should store them in a dedicated directory (e.g., `pipelines/`) and pass that path to CMake when building `pipebench` via the `-DPIPES_DIR=<pipelines_dir>` option.

3) Example end-to-end

```bash
# 1) Generate pipeline
cmake -S . -B build -DBUILD_TARGET=pipegen && cmake --build build -j
./build/bin/pipegen -o pipelines/pipeline_0_1 --dag-seed 123 --schedule-seed 456

# 2) Build benchmarks
cmake -S . -B build -DBUILD_TARGET=pipebench -DPIPES_DIR=pipelines && cmake --build build -j

# 3) Run an existing pipeline-specific benchmark (if present)
./build/bin/pipebench_pipeline_0_1
```