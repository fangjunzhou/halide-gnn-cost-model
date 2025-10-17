set(PIPELINE_DIR "${CMAKE_SOURCE_DIR}/pipelines/example")
set(PIPELINE_LIB_PATH "${PIPELINE_DIR}/pipeline${CMAKE_STATIC_LIBRARY_SUFFIX}")

if(NOT EXISTS "${PIPELINE_LIB_PATH}")
  message(FATAL_ERROR "Pipeline library not found at: ${PIPELINE_LIB_PATH}")
endif()
if(NOT EXISTS "${PIPELINE_DIR}/pipeline.h")
  message(
    FATAL_ERROR
      "Pipeline header directory not found at: ${PIPELINE_DIR}/pipeline.h")
endif()
message(STATUS "Using pipeline library at: ${PIPELINE_LIB_PATH}")

add_library(pipeline_lib STATIC IMPORTED)
set_target_properties(
  pipeline_lib PROPERTIES IMPORTED_LOCATION "${PIPELINE_LIB_PATH}"
                          INTERFACE_INCLUDE_DIRECTORIES "${PIPELINE_DIR}")

add_executable(pipebench main.cpp)
target_link_libraries(pipebench PRIVATE spdlog::spdlog)
target_link_libraries(pipebench PRIVATE pipeline_lib)
target_link_libraries(pipebench PRIVATE benchmark::benchmark)

install(TARGETS pipebench DESTINATION bin)
