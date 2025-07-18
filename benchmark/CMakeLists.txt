# Benchmark
add_executable(bench_gemm
    bench_gemm.cpp
)
add_executable(bench_brgemm
    bench_brgemm.cpp
)
add_executable(bench_unary_jit
    bench_unary_jit.cpp
)

target_link_libraries(bench_gemm PRIVATE mini_jit)
target_link_libraries(bench_brgemm PRIVATE mini_jit)
target_link_libraries(bench_unary_jit PRIVATE mini_jit)
