cmake_minimum_required(VERSION 3.14)
project(TenGenProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output binary files into bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Subprojects
add_subdirectory(TenGen)
enable_testing()
add_subdirectory(TenGenTests)
