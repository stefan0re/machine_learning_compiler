cmake_minimum_required(VERSION 3.10)
project(TensorCompiler)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_ASM_COMPILER as)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "Current source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(./benchmarks/microbenchmarks)
