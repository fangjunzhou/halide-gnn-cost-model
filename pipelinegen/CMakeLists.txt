# Main executable
add_executable(pipegen ./main.cpp)
target_link_libraries(pipegen spdlog::spdlog)

install(TARGETS pipegen DESTINATION bin)
