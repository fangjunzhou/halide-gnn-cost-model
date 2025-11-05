# Main executable
file(GLOB_RECURSE PIPELINEGEN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
add_executable(pipegen ${PIPELINEGEN_SOURCES})
target_link_libraries(pipegen PRIVATE spdlog::spdlog)
target_link_libraries(pipegen PRIVATE Halide::Halide)
target_link_libraries(pipegen PRIVATE nlohmann_json::nlohmann_json)

install(TARGETS pipegen DESTINATION bin)
