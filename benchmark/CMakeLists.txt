# Unary-Benchmark
add_executable(mini_jit_unary_benchmark
    driver.cpp
    ../test/test_utils.cpp
)
target_link_libraries(mini_jit_unary_benchmark PRIVATE mini_jit)
