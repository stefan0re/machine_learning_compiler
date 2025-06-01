cmake_minimum_required(VERSION 3.14)
project(TenGenTests)

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.3
)

FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(
  fastor
  GIT_REPOSITORY https://github.com/romeric/Fastor.git
  GIT_TAG master  
)

FetchContent_MakeAvailable(fastor)

enable_testing()

add_subdirectory(mini_jit/instructions)
add_subdirectory(mini_jit/generator)