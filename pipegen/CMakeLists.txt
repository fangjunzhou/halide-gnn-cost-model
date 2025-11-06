# Collect source files for the main pipegen binary, EXCLUDING test_*.cpp files
file(GLOB PIPEGEN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(FILTER PIPEGEN_SOURCES EXCLUDE REGEX ".*/test_.*\\.cpp$")

# Primary generator binary
add_executable(pipegen ${PIPEGEN_SOURCES})
target_link_libraries(pipegen PRIVATE spdlog::spdlog Halide::Halide nlohmann_json::nlohmann_json)
# Make vendored argparse header available to pipegen
target_include_directories(pipegen PRIVATE ${CMAKE_SOURCE_DIR}/third_party/argparse/include
                                         ${CMAKE_CURRENT_SOURCE_DIR})

# AST test: simple JSON AST test for a single expression
add_executable(ast_test test_ast.cpp)
target_include_directories(ast_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ast_test PRIVATE Halide::Halide nlohmann_json::nlohmann_json)

# JSON AST test: builds a random expression and writes JSON AST
add_executable(ast_json_test test_json_ast.cpp)
target_include_directories(ast_json_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ast_json_test PRIVATE Halide::Halide nlohmann_json::nlohmann_json)

install(TARGETS pipegen DESTINATION bin)