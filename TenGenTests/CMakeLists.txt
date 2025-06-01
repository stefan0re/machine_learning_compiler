cmake_minimum_required(VERSION 3.14)
project(TenGenTests)

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.3
)

FetchContent_MakeAvailable(Catch2)

# Fetch xtl first
FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
  GIT_TAG master  
)

FetchContent_MakeAvailable(xtl)

# Fetch xtensor
FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
  GIT_TAG master  
)

FetchContent_MakeAvailable(xtensor)

enable_testing()

add_subdirectory(mini_jit/instructions)
add_subdirectory(mini_jit/generator)
add_subdirectory(tensor)