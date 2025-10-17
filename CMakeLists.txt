cmake_minimum_required(VERSION 3.29.2)

project(halide_gnn_cost_model VERSION 0.1.0)

# Enable compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Project binary and library output directory.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/lib
    CACHE STRING "Library Output" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin
    CACHE STRING "Binary Output" FORCE)

# Optimization level.
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies.
find_package(spdlog REQUIRED)
find_package(Halide REQUIRED)

# Build targets.
set(BUILD_TARGET
    "pipegen"
    CACHE STRING "Which pipeline executable to build")
set_property(CACHE BUILD_TARGET PROPERTY STRINGS pipegen pipebench)

# Build main project.
if(BUILD_TARGET STREQUAL "pipebench")
  add_subdirectory(./pipebench)
  message(STATUS "Building pipebench")
elseif(BUILD_TARGET STREQUAL "pipegen")
  add_subdirectory(./pipegen)
  message(STATUS "Building pipegen")
else()
  message(FATAL_ERROR "Unknown BUILD_TARGET: ${BUILD_TARGET}")
endif()
