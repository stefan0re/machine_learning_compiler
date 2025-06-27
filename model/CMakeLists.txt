cmake_minimum_required(VERSION 3.20)
project(ModelProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_NAME run_model)


set(SOURCE_FILES
    model.cpp
    ../src/tensor/tensor.cpp
    ../src/mini_jit/generator/Brgemm.cpp
    ../src/mini_jit/generator/Util.cpp
    ../src/mini_jit/backend/Kernel.cpp
    ../src/mini_jit/instructions/neon.cpp
    ../src/mini_jit/instructions/base.cpp
    ../src/mini_jit/generator/Unary.cpp
    ../src/mini_jit/include/gemm_ref.cpp
    ../src/einsum/backend/TensorOperation.cpp
    ../src/einsum/trees/einsum_trees.cpp
)

add_executable(${EXECUTABLE_NAME} ${SOURCE_FILES})

include_directories(
    ../src/tensor
    ../src/mini_jit/include
    ../src/einsum/backend
    ../src/einsum/trees
)
