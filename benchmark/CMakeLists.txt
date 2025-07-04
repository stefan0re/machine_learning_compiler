# Unary-Benchmark
add_executable(bench_gemm
    bench_gemm.cpp
)
target_link_libraries(bench_gemm PRIVATE mini_jit)
