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

add_executable( bench_tree
    bench_tree.cpp
)

add_executable( check_ten_op_unary
    check_ten_op_unary.cpp
)

add_executable( check_iris_model
    check_iris_model.cpp
)

target_link_libraries(bench_gemm PRIVATE mini_jit)
target_link_libraries(bench_brgemm PRIVATE mini_jit)
target_link_libraries(bench_unary_jit PRIVATE mini_jit)
target_link_libraries(bench_ten_op_backend PRIVATE einsum)
target_link_libraries(bench_ten_op_optimize PRIVATE einsum)
target_link_libraries(bench_tree PRIVATE einsum)
target_link_libraries( check_ten_op_unary PRIVATE einsum)
target_link_libraries( check_iris_model PRIVATE einsum)
