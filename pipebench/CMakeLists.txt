if(NOT DEFINED PIPES_DIR)
  # Default to the repository's pipelines directory. External callers can
  # still override via -DPIPES_DIR=/some/path when invoking cmake; we avoid
  # declaring a cached default here so the default won't be written into the
  # CMake cache unintentionally.
  set(PIPES_DIR "${CMAKE_SOURCE_DIR}/resources/pipelines")
endif()

# Normalize PIPES_DIR to an absolute path. If the user provided a relative
# path, interpret it relative to the top-level source dir so paths like
# "resources/pipelines" resolve to the project-root resources directory.
if(NOT IS_ABSOLUTE "${PIPES_DIR}")
  get_filename_component(PIPES_DIR "${CMAKE_SOURCE_DIR}/${PIPES_DIR}" ABSOLUTE)
else()
  get_filename_component(PIPES_DIR "${PIPES_DIR}" ABSOLUTE)
endif()

# Optionally allow specifying a single pipeline directory. When provided,
# PIPE_DIR takes precedence and only that pipeline will be built.
if(NOT DEFINED PIPE_DIR)
  # Allow specifying a single pipeline directory. Do not declare this as a
  # cached variable so it won't persist in CMakeCache.txt unexpectedly.
  set(PIPE_DIR "")
endif()

# Find all pipeline directories named pipeline_*
if(DEFINED PIPE_DIR AND NOT "${PIPE_DIR}" STREQUAL "")
  # Single-pipeline mode: PIPE_DIR must contain the expected pipeline files.
  if(EXISTS "${PIPE_DIR}/pipeline.h")
    set(SINGLE_PIPELINE_LIB_PATH "${PIPE_DIR}/pipeline${CMAKE_STATIC_LIBRARY_SUFFIX}")
    if(EXISTS "${SINGLE_PIPELINE_LIB_PATH}")
      message(STATUS "PIPE_DIR specified; building single pipeline at: ${PIPE_DIR}")
      # Resolve PIPE_DIR relative to the project source root if it's
      # a relative path, then use the absolute result as the sole entry.
      if(IS_ABSOLUTE "${PIPE_DIR}")
        get_filename_component(ABS_PIPE_DIR "${PIPE_DIR}" ABSOLUTE)
      else()
        get_filename_component(ABS_PIPE_DIR "${CMAKE_SOURCE_DIR}/${PIPE_DIR}" ABSOLUTE)
      endif()
      set(PIPELINE_DIRS "${ABS_PIPE_DIR}")
    else()
      message(FATAL_ERROR "PIPE_DIR set to ${PIPE_DIR} but pipeline library missing at: ${SINGLE_PIPELINE_LIB_PATH}")
    endif()
  else()
    message(FATAL_ERROR "PIPE_DIR set to ${PIPE_DIR} but pipeline.h not found in that directory")
  endif()
else()
  # Normal mode: discover all directories under PIPES_DIR. Use absolute
  # paths so later logic can assume entries are absolute.
  file(
    GLOB PIPELINE_DIRS
    "${PIPES_DIR}/*")
  if(NOT PIPELINE_DIRS)
    message(FATAL_ERROR "No directories found in ${PIPES_DIR}")
  endif()
endif()

message(STATUS "Discovered pipeline dirs: ${PIPELINE_DIRS}")

foreach(pdir IN LISTS PIPELINE_DIRS)
  # Entries in PIPELINE_DIRS are absolute paths; treat them directly.
  set(PIPELINE_DIR "${pdir}")

  set(PIPELINE_LIB_PATH "${PIPELINE_DIR}/pipeline${CMAKE_STATIC_LIBRARY_SUFFIX}")

  if(NOT EXISTS "${PIPELINE_LIB_PATH}")
    message(FATAL_ERROR "Pipeline library not found at: ${PIPELINE_LIB_PATH}")
  endif()
  if(NOT EXISTS "${PIPELINE_DIR}/pipeline.h")
    message(FATAL_ERROR "Pipeline header not found at: ${PIPELINE_DIR}/pipeline.h")
  endif()

  message(STATUS "Using pipeline library at: ${PIPELINE_LIB_PATH}")

  # Create an imported target per pipeline directory. Target names must be
  # unique; use the directory basename as a suffix.
  get_filename_component(PIPE_BASENAME "${PIPELINE_DIR}" NAME)
  # Replace any characters that might be problematic in a target name.
  string(REPLACE "." "_" PIPE_BASENAME_SAFE "${PIPE_BASENAME}")
  string(REPLACE "/" "_" PIPE_SUFFIX "${PIPE_BASENAME_SAFE}")
  set(PIPELINE_LIB_TARGET "pipeline_lib_${PIPE_SUFFIX}")
  add_library(${PIPELINE_LIB_TARGET} STATIC IMPORTED)
  set_target_properties(
    ${PIPELINE_LIB_TARGET}
    PROPERTIES IMPORTED_LOCATION "${PIPELINE_LIB_PATH}"
               INTERFACE_INCLUDE_DIRECTORIES "${PIPELINE_DIR}")

  # Add executable specific to this pipeline.
  set(PIPEBENCH_EXEC "pipebench_${PIPE_SUFFIX}")
  add_executable(${PIPEBENCH_EXEC} main.cpp)
  target_link_libraries(${PIPEBENCH_EXEC} PRIVATE Halide::Halide)
  target_link_libraries(${PIPEBENCH_EXEC} PRIVATE spdlog::spdlog)
  target_link_libraries(${PIPEBENCH_EXEC} PRIVATE ${PIPELINE_LIB_TARGET})
  target_link_libraries(${PIPEBENCH_EXEC} PRIVATE benchmark::benchmark)

  install(TARGETS ${PIPEBENCH_EXEC} DESTINATION bin)
endforeach()
