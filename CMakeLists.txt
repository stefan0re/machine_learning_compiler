cmake_minimum_required(VERSION 3.10)
project(mlc)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_ASM_COMPILER as)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CTest)

add_subdirectory(./src/jiter)

enable_testing()
add_subdirectory(./test)


