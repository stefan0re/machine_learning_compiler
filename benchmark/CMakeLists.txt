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
add_executable(bench_ten_op_backend
    bench_ten_op_backend.cpp
)

add_executable( bench_ten_op_optimize
    bench_ten_op_optimize.cpp
)

target_link_libraries(bench_gemm PRIVATE mini_jit)
target_link_libraries(bench_brgemm PRIVATE mini_jit)
target_link_libraries(bench_unary_jit PRIVATE mini_jit)
target_link_libraries(bench_ten_op_backend PRIVATE einsum)
target_link_libraries(bench_ten_op_optimize PRIVATE einsum)
