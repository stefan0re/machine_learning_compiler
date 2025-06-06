cmake_minimum_required(VERSION 3.27.7)
project(TenGenLib)

# All headers are in this folder or subfolders
add_library(TenGen INTERFACE)

# Make headers available to other targets
target_include_directories(TenGen INTERFACE ${CMAKE_SOURCE_DIR})

target_link_libraries(TenGen INTERFACE OpenMP::OpenMP_CXX)

# Optional: if you want to organize includes via TenGen.h
install(FILES TenGen.h DESTINATION include)
