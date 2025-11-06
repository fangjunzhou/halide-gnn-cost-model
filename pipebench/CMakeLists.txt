if(NOT DEFINED PIPES_DIR)
  # Default to the repository's pipelines directory, but allow external
  # callers (e.g. scripts) to override via -DPIPES_DIR=/some/path when
  # invoking cmake.
  set(PIPES_DIR "${CMAKE_SOURCE_DIR}/pipelines" CACHE PATH "Directory containing pipeline_* dirs")
endif()

# Find all pipeline directories named pipeline_*
file(
  GLOB PIPELINE_DIRS
  RELATIVE "${PIPES_DIR}"
  "${PIPES_DIR}/pipeline_*")
if(NOT PIPELINE_DIRS)
  message(FATAL_ERROR "No pipeline_* directories found in ${PIPES_DIR}")
endif()

message(STATUS "Discovered pipeline dirs: ${PIPELINE_DIRS}")

foreach(pdir IN LISTS PIPELINE_DIRS)
  set(PIPELINE_DIR "${PIPES_DIR}/${pdir}")
  set(PIPELINE_LIB_PATH
      "${PIPELINE_DIR}/pipeline${CMAKE_STATIC_LIBRARY_SUFFIX}")

  if(NOT EXISTS "${PIPELINE_LIB_PATH}")
    message(FATAL_ERROR "Pipeline library not found at: ${PIPELINE_LIB_PATH}")
  endif()
  if(NOT EXISTS "${PIPELINE_DIR}/pipeline.h")
    message(
      FATAL_ERROR "Pipeline header not found at: ${PIPELINE_DIR}/pipeline.h")
  endif()

  message(STATUS "Using pipeline library at: ${PIPELINE_LIB_PATH}")

  # Create an imported target per pipeline directory. Target names must be
  # unique; use the directory name suffix.
  string(REPLACE "/" "_" PIPE_SUFFIX "${pdir}")
  set(PIPELINE_LIB_TARGET "pipeline_lib_${PIPE_SUFFIX}")
  add_library(${PIPELINE_LIB_TARGET} STATIC IMPORTED)
  set_target_properties(
    ${PIPELINE_LIB_TARGET}
    PROPERTIES IMPORTED_LOCATION "${PIPELINE_LIB_PATH}"
               INTERFACE_INCLUDE_DIRECTORIES "${PIPELINE_DIR}")

  # Add executable specific to this pipeline.
  set(PIPEBENCH_EXEC "pipebench_${PIPE_SUFFIX}")
  add_executable(${PIPEBENCH_EXEC} main.cpp)
  target_link_libraries(${PIPEBENCH_EXEC} PRIVATE spdlog::spdlog)
  target_link_libraries(${PIPEBENCH_EXEC} PRIVATE ${PIPELINE_LIB_TARGET})
  target_link_libraries(${PIPEBENCH_EXEC} PRIVATE benchmark::benchmark)

  # Determine numeric pipeline index from directory name (e.g. pipeline_3 -> 3).
  # Use CMake's MATCHES operator to capture the trailing number into
  # CMAKE_MATCH_1 so it's robust across CMake versions.
  if("${pdir}" MATCHES ".*_([0-9]+)$")
    set(PIPE_IDX_MATCH "${CMAKE_MATCH_1}")
  else()
    message(
      FATAL_ERROR "Failed to determine numeric pipeline index from ${pdir}")
  endif()

  install(TARGETS ${PIPEBENCH_EXEC} DESTINATION bin)
endforeach()
