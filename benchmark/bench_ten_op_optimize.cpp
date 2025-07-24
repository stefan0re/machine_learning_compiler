
#include <chrono>
#include <iostream>

#include "../src/einsum/backend/TensorOperation.h"
#include "../src/tensor/tensor.h"

using namespace einsum::backend;

#define DEBUG
/**
 *  dim_types	( M, N, K )
    exec_types	( Seq, Seq, Seq )
    dim_sizes	( 1600, 1600, 1600 )
    strides_in0	( 1, 0, 1600 )
    strides_in1	( 0, 1600, 1 )
    strides_out	( 1, 1600, 0 )
 */
void example1_ref(float* in0,
                  float* in1,
                  float* out) {
    for (size_t n = 0; n < 1600; n++) {
        for (size_t m = 0; m < 1600; m++) {
            for (size_t k = 0; k < 1600; k++) {
                // compute the index for in0, in1 and out
                size_t idx_in0 = k * 1600 + m;
                size_t idx_in1 = n * 1600 + k;
                size_t idx_out = n * 1600 + m;

                // perform the operation
                out[idx_out] += in0[idx_in0] * in1[idx_in1];
            }
        }
    }
}

void first_example() {
    // Testing first example

    std::cout << "Running first example with optimizations..." << std::endl;
    TensorOperation tensor_op;

    std::vector<TensorOperation::dim_t> i_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k};
    std::vector<TensorOperation::exec_t> i_exec_types = {TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq};

    std::vector<int64_t> i_dim_sizes = {1600, 1600, 1600};
    std::vector<int64_t> i_strides_in0 = {1, 0, 1600};
    std::vector<int64_t> i_strides_in1 = {0, 1600, 1};
    std::vector<int64_t> i_strides_out = {1, 1600, 0};

    std::span<TensorOperation::dim_t> i_dim_types_span(i_dim_types);
    std::span<TensorOperation::exec_t> i_exec_types_span(i_exec_types);
    std::span<int64_t> i_dim_sizes_span(i_dim_sizes);
    std::span<int64_t> i_strides_in0_span(i_strides_in0);
    std::span<int64_t> i_strides_in1_span(i_strides_in1);
    std::span<int64_t> i_strides_out_span(i_strides_out);

    tensor_op.setup(TensorOperation::dtype_t::fp32,
                    TensorOperation::prim_t::none,
                    TensorOperation::prim_t::gemm,
                    TensorOperation::prim_t::none,
                    i_dim_types_span,
                    i_exec_types_span,
                    i_dim_sizes_span,
                    i_strides_in0_span,
                    i_strides_in1_span,
                    i_strides_out_span);

    tensor_op.optimize();

    tensor_op.compile();

    int64_t size_in0 = 1600 * 1600;
    int64_t size_in1 = 1600 * 1600;
    int64_t size_out = 1600 * 1600;

    float* tensor_in0 = new float[size_in0];
    float* tensor_in1 = new float[size_in1];
    float* tensor_out = new float[size_out];
    float* tensor_out_ref = new float[size_out];

    // Initialize input tensors
    srand(time(NULL));
    for (size_t i = 0; i < size_in0; i++) {
        tensor_in0[i] = (float)drand48();
    }
    for (size_t i = 0; i < size_in1; i++) {
        tensor_in1[i] = (float)drand48();
    }
    for (size_t i = 0; i < size_out; i++) {
        tensor_out[i] = 0.0f;
        tensor_out_ref[i] = 0.0f;
    }

    // execute refernece
    example1_ref(tensor_in0, tensor_in1, tensor_out_ref);

    // execute optimized tensor operation
    tensor_op.execute(tensor_in0, tensor_in1, tensor_out);

    // verify results
    double error = 0.0;
    for (size_t i = 0; i < size_out; i++) {
        error += std::abs(tensor_out[i] - tensor_out_ref[i]);
    }

    std::cout << "  Total error first example: " << error << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 10; i++) {
        tensor_op.execute(tensor_in0, tensor_in1, tensor_out);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "  Execution first for third example: " << elapsed.count() << " seconds" << std::endl;
    double gflops = tensor_op.get_flops_count() / elapsed.count() / 1e9;
    gflops *= 10;
    std::cout << "  GFLOPS for first example: " << gflops << std::endl;

    // Clean up
    delete[] tensor_in0;
    delete[] tensor_in1;
    delete[] tensor_out;
    delete[] tensor_out_ref;

    // Print debug information
#ifdef DEBUG
    // print dim sizes types and strides
    std::cout << "Dimension types: ";
    for (const auto& dim_type : tensor_op._dim_types) {
        std::cout << static_cast<int>(dim_type) << " ";
    }
    std::cout << std::endl;
    std::cout << "Dimension sizes: ";
    for (const auto& dim_size : tensor_op._dim_sizes) {
        std::cout << dim_size << " ";
    }
    std::cout << std::endl;
    std::cout << "Execution types: ";
    for (const auto& exec_type : tensor_op._exec_types) {
        std::cout << static_cast<int>(exec_type) << " ";
    }

    std::cout << std::endl;
    std::cout << "Strides in0: ";
    for (const auto& stride : tensor_op._strides_in0) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "Strides in1: ";
    for (const auto& stride : tensor_op._strides_in1) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "Strides out: ";
    for (const auto& stride : tensor_op._strides_out) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;

#endif
}

/**
 *  dim_types	( M0, M1, N0, N1, K0, K1 )
    exec_types	( Seq, Seq, Seq, Seq, Seq, Seq )
    dim_sizes	( 64, 25, 64, 25, 64, 25 )
    strides_in0	( 25, 1, 0, 0, 40000, 1600 )
    strides_in1	( 0, 0, 40000, 1600, 25, 1 )
    strides_out	( 25, 1, 40000, 1600, 0, 0 )
 */
void example2_ref(float* in0,
                  float* in1,
                  float* out) {
    for (size_t M0 = 0; M0 < 64; M0++) {
        for (size_t M1 = 0; M1 < 25; M1++) {
            for (size_t N0 = 0; N0 < 64; N0++) {
                for (size_t N1 = 0; N1 < 25; N1++) {
                    for (size_t K0 = 0; K0 < 64; K0++) {
                        for (size_t K1 = 0; K1 < 25; K1++) {
                            // compute the index for in0, in1 and out
                            size_t idx_in0 = M1 + M0 * 25 + K1 * 1600 + K0 * 40000;
                            size_t idx_in1 = K1 + K0 * 25 + N1 * 1600 + N0 * 40000;
                            size_t idx_out = M1 + M0 * 25 + N1 * 1600 + N0 * 40000;

                            // perform the operation
                            out[idx_out] += in0[idx_in0] * in1[idx_in1];
                        }
                    }
                }
            }
        }
    }
}

void second_example() {
    std::cout << "Running second example with optimizations..." << std::endl;
    TensorOperation tensor_op;

    std::vector<TensorOperation::dim_t> i_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k,
                                                       TensorOperation::dim_t::k};
    std::vector<TensorOperation::exec_t> i_exec_types = {TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq};

    std::vector<int64_t> i_dim_sizes = {64, 25, 64, 25, 64, 25};
    std::vector<int64_t> i_strides_in0 = {25, 1, 0, 0, 40000, 1600};
    std::vector<int64_t> i_strides_in1 = {0, 0, 40000, 1600, 25, 1};
    std::vector<int64_t> i_strides_out = {25, 1, 40000, 1600, 0, 0};

    std::span<TensorOperation::dim_t> i_dim_types_span(i_dim_types);
    std::span<TensorOperation::exec_t> i_exec_types_span(i_exec_types);
    std::span<int64_t> i_dim_sizes_span(i_dim_sizes);
    std::span<int64_t> i_strides_in0_span(i_strides_in0);
    std::span<int64_t> i_strides_in1_span(i_strides_in1);
    std::span<int64_t> i_strides_out_span(i_strides_out);

    tensor_op.setup(TensorOperation::dtype_t::fp32,
                    TensorOperation::prim_t::none,
                    TensorOperation::prim_t::gemm,
                    TensorOperation::prim_t::none,
                    i_dim_types_span,
                    i_exec_types_span,
                    i_dim_sizes_span,
                    i_strides_in0_span,
                    i_strides_in1_span,
                    i_strides_out_span);

    tensor_op.optimize();

    tensor_op.compile();

    int64_t size_in0 = 64 * 25 * 64 * 25;
    int64_t size_in1 = 64 * 25 * 64 * 25;
    int64_t size_out = 64 * 25 * 64 * 25;
    float* tensor_in0 = new float[size_in0];
    float* tensor_in1 = new float[size_in1];
    float* tensor_out = new float[size_out];
    float* tensor_out_ref = new float[size_out];

    // Initialize input tensors
    srand(time(NULL));
    for (size_t i = 0; i < size_in0; i++) {
        tensor_in0[i] = (float)drand48();
    }
    for (size_t i = 0; i < size_in1; i++) {
        tensor_in1[i] = (float)drand48();
    }
    for (size_t i = 0; i < size_out; i++) {
        tensor_out[i] = 0.0f;
        tensor_out_ref[i] = 0.0f;
    }

    // execute reference
    example2_ref(tensor_in0, tensor_in1, tensor_out_ref);

    // execute optimized tensor operation
    tensor_op.execute(tensor_in0, tensor_in1, tensor_out);

    // verify results
    double error = 0.0;
    for (size_t i = 0; i < size_out; i++) {
        error += std::abs(tensor_out[i] - tensor_out_ref[i]);
    }
    std::cout << "  Total error second example: " << error << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 10; i++) {
        tensor_op.execute(tensor_in0, tensor_in1, tensor_out);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "  Execution time for second example: " << elapsed.count() << " seconds" << std::endl;
    double gflops = tensor_op.get_flops_count() / elapsed.count() / 1e9;
    gflops *= 10;
    std::cout << "  GFLOPS for second example: " << gflops << std::endl;

    // Clean up
    delete[] tensor_in0;
    delete[] tensor_in1;
    delete[] tensor_out;
    delete[] tensor_out_ref;

    // Print debug information
#ifdef DEBUG
    // print dim sizes types and strides
    std::cout << "Dimension types: ";
    for (const auto& dim_type : tensor_op._dim_types) {
        std::cout << static_cast<int>(dim_type) << " ";
    }
    std::cout << std::endl;
    std::cout << "Dimension sizes: ";
    for (const auto& dim_size : tensor_op._dim_sizes) {
        std::cout << dim_size << " ";
    }
    std::cout << std::endl;
    std::cout << "Execution types: ";
    for (const auto& exec_type : tensor_op._exec_types) {
        std::cout << static_cast<int>(exec_type) << " ";
    }

    std::cout << std::endl;
    std::cout << "Strides in0: ";
    for (const auto& stride : tensor_op._strides_in0) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "Strides in1: ";
    for (const auto& stride : tensor_op._strides_in1) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    std::cout << "Strides out: ";
    for (const auto& stride : tensor_op._strides_out) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;

#endif
}

int main() {
    std::cout << "Benchmarking Tensor contraction with optimization ..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    first_example();
    std::cout << "----------------------------------------" << std::endl;
    second_example();
    std::cout << "----------------------------------------" << std::endl;

    return EXIT_SUCCESS;
}