cmake_minimum_required(VERSION 3.30.8)
project(TenGenProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output binary files into bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# find OpenMP
find_package(OpenMP REQUIRED)

add_subdirectory(src/mini_jit)
add_subdirectory(src/einsum)
# add_subdirectory(model)
add_subdirectory(benchmark)

# enable_testing()

# Subprojects
# add_subdirectory(test)

