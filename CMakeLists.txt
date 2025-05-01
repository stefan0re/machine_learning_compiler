cmake_minimum_required(VERSION 3.10)
project(TensorCompiler)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_ASM_COMPILER as)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CTest)
enable_testing()

add_subdirectory(./benchmarks/microbenchmarks_gp/throughput)
add_subdirectory(./benchmarks/microbenchmarks_gp/latency)
add_subdirectory(./benchmarks/microbenchmarks_neon)
add_subdirectory(./assembly_examples/task_2)
add_subdirectory(./assembly_examples/task_3/microkernel)
