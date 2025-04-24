cmake_minimum_required(VERSION 3.10)
project(TensorCompiler)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_ASM_COMPILER as)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(./benchmarks/microbenchmarks)
add_subdirectory(./assembly_examples/task_2)
