cmake_minimum_required(VERSION 3.14)
project(TenGenLib)

# All headers are in this folder or subfolders
add_library(TenGen INTERFACE)

# Make headers available to other targets
target_include_directories(TenGen INTERFACE ${CMAKE_SOURCE_DIR})

# Optional: if you want to organize includes via TenGen.h
install(FILES TenGen.h DESTINATION include)
