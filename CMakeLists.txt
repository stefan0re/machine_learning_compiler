cmake_minimum_required(VERSION 3.27.7)
project(TenGenProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output binary files into bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# find OpenMP
find_package(OpenMP REQUIRED)

enable_testing()

# Subprojects
add_subdirectory(TenGen)
add_subdirectory(TenGenTests)

