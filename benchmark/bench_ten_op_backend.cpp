
#include <chrono>
#include <iostream>

#include "../src/einsum/backend/TensorOperation.h"

using namespace einsum::backend;

/**
 * dtype	            FP32
    prim_first_touch	None
    prim_main	        GEMM
    prim_last_touch	    None
    dim_types	        ( M, N, K, M, N, K )
    exec_types	        ( Seq, Seq, Seq, Prim, Prim, Prim )
    dim_sizes	        ( 32, 32, 8, 32, 32, 32 )
    strides_in0	        ( 8192, 0, 1024, 1, 0, 32 )
    strides_in1	        ( 0, 8192, 1024, 0, 32, 1 )
    strides_out	        ( 32768, 1024, 0, 1, 32, 0 )
 */
void example1_2_ref(float* in0,
                    float* in1,
                    float* out) {
    for (size_t M = 0; M < 32; M++) {
        for (size_t N = 0; N < 32; N++) {
            for (size_t K = 0; K < 8; K++) {
                for (size_t m = 0; m < 32; m++) {
                    for (size_t n = 0; n < 32; n++) {
                        for (size_t k = 0; k < 32; k++) {
                            // compute the index for in0, in1 and out
                            size_t idx_in0 = M * 8192 + K * 1024 + m * 1 + n * 0 + k * 32;
                            size_t idx_in1 = N * 8192 + K * 1024 + m * 0 + n * 32 + k * 1;
                            size_t idx_out = M * 32768 + N * 1024 + K * 0 + m * 1 + n * 32 + k * 0;

                            // perform the operation
                            out[idx_out] += in0[idx_in0] * in1[idx_in1];
                        }
                    }
                }
            }
        }
    }
}

void first_example() {
    std::cout << "Running first example..." << std::endl;

    TensorOperation tensor_op;

    std::vector<TensorOperation::dim_t> i_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k,
                                                       TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k};
    std::vector<TensorOperation::exec_t> i_exec_types = {TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim};

    std::vector<int64_t> i_dim_sizes = {32, 32, 8, 32, 32, 32};
    std::vector<int64_t> i_strides_in0 = {8192, 0, 1024, 1, 0, 32};
    std::vector<int64_t> i_strides_in1 = {0, 8192, 1024, 0, 32, 1};
    std::vector<int64_t> i_strides_out = {32768, 1024, 0, 1, 32, 0};

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

    tensor_op.compile();

    // create in0, in1 and out tensor
    int64_t size_in0 = 32 * 8 * 32 * 32;
    int64_t size_in1 = 32 * 8 * 32 * 32;
    int64_t size_out = 32 * 32 * 32 * 32;

    float* tensor_in0 = new float[size_in0];
    float* tensor_in1 = new float[size_in1];
    float* tensor_out = new float[size_out];
    float* tensor_out_ref = new float[size_out];

    // initialize input tensors
    srand(42);
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
    example1_2_ref(tensor_in0, tensor_in1, tensor_out_ref);
    // execute TenOp
    tensor_op.execute(tensor_in0, tensor_in1, tensor_out);

    // verify results
    double error = 0.0;
    for (size_t i = 0; i < size_out; i++) {
        error += std::abs(tensor_out[i] - tensor_out_ref[i]);
    }
    std::cout << "  Total error first example: " << error << std::endl;

    // benchmark execution time
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 200; i++) {
        tensor_op.execute(tensor_in0, tensor_in1, tensor_out);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "  Execution time for third example: " << elapsed.count() << " seconds" << std::endl;
    double gflops = tensor_op.get_flops_count() / elapsed.count() / 1e9;
    gflops *= 200;
    std::cout << "  GFLOPS for third example: " << gflops << std::endl;

    // cleanup
    delete[] tensor_in0;
    delete[] tensor_in1;
    delete[] tensor_out;
    delete[] tensor_out_ref;
}

/**
 * dtype	            FP32
    prim_first_touch	None
    prim_main	        GEMM
    prim_last_touch	    None
    dim_types	        ( M, N, K, M, N, K )
    exec_types	        ( Seq, Seq, Prim, Prim, Prim, Prim )
    dim_sizes	        ( 32, 32, 8, 32, 32, 32 )
    strides_in0	        ( 8192, 0, 1024, 1, 0, 32 )
    strides_in1	        ( 0, 8192, 1024, 0, 32, 1 )
    strides_out	        ( 32768, 1024, 0, 1, 32, 0 )
 */
void second_example() {
    std::cout << "Running second example..." << std::endl;

    TensorOperation tensor_op;

    std::vector<TensorOperation::dim_t> i_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k,
                                                       TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k};
    std::vector<TensorOperation::exec_t> i_exec_types = {TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim};

    std::vector<int64_t> i_dim_sizes = {32, 32, 8, 32, 32, 32};
    std::vector<int64_t> i_strides_in0 = {8192, 0, 1024, 1, 0, 32};
    std::vector<int64_t> i_strides_in1 = {0, 8192, 1024, 0, 32, 1};
    std::vector<int64_t> i_strides_out = {32768, 1024, 0, 1, 32, 0};

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

    tensor_op.compile();

    // create in0, in1 and out tensor
    int64_t size_in0 = 32 * 8 * 32 * 32;
    int64_t size_in1 = 32 * 8 * 32 * 32;
    int64_t size_out = 32 * 32 * 32 * 32;

    float* tensor_in0 = new float[size_in0];
    float* tensor_in1 = new float[size_in1];
    float* tensor_out = new float[size_out];
    float* tensor_out_ref = new float[size_out];

    // initialize input tensors
    srand(42);
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
    example1_2_ref(tensor_in0, tensor_in1, tensor_out_ref);
    // execute TenOp
    tensor_op.execute(tensor_in0, tensor_in1, tensor_out);

    // verify results
    double error = 0.0;
    for (size_t i = 0; i < size_out; i++) {
        error += std::abs(tensor_out[i] - tensor_out_ref[i]);
    }
    std::cout << "  Total error second example: " << error << std::endl;

    // benchmark execution time
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 200; i++) {
        tensor_op.execute(tensor_in0, tensor_in1, tensor_out);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "  Execution time for third example: " << elapsed.count() << " seconds" << std::endl;
    double gflops = tensor_op.get_flops_count() / elapsed.count() / 1e9;
    gflops *= 200;
    std::cout << "  GFLOPS for third example: " << gflops << std::endl;

    // cleanup
    delete[] tensor_in0;
    delete[] tensor_in1;
    delete[] tensor_out;
    delete[] tensor_out_ref;
}

/**
 * dtype	            FP32
    prim_first_touch	None
    prim_main	        GEMM
    prim_last_touch	    RELU
    dim_types	        ( M, N, K, M, N, K )
    exec_types	        ( Seq, Seq, Prim, Prim, Prim, Prim )
    dim_sizes	        ( 32, 32, 8, 32, 32, 32 )
    strides_in0	        ( 8192, 0, 1024, 1, 0, 32 )
    strides_in1	        ( 0, 8192, 1024, 0, 32, 1 )
    strides_out	        ( 32768, 1024, 0, 1, 32, 0 )
 */
void third_example() {
    // Testing first example

    std::cout << "Running third example with ReLU activation..." << std::endl;

    TensorOperation tensor_op;

    std::vector<TensorOperation::dim_t> i_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k,
                                                       TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k};
    std::vector<TensorOperation::exec_t> i_exec_types = {TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim};

    std::vector<int64_t> i_dim_sizes = {32, 32, 8, 32, 32, 32};
    std::vector<int64_t> i_strides_in0 = {8192, 0, 1024, 1, 0, 32};
    std::vector<int64_t> i_strides_in1 = {0, 8192, 1024, 0, 32, 1};
    std::vector<int64_t> i_strides_out = {32768, 1024, 0, 1, 32, 0};

    std::span<TensorOperation::dim_t> i_dim_types_span(i_dim_types);
    std::span<TensorOperation::exec_t> i_exec_types_span(i_exec_types);
    std::span<int64_t> i_dim_sizes_span(i_dim_sizes);
    std::span<int64_t> i_strides_in0_span(i_strides_in0);
    std::span<int64_t> i_strides_in1_span(i_strides_in1);
    std::span<int64_t> i_strides_out_span(i_strides_out);

    tensor_op.setup(TensorOperation::dtype_t::fp32,
                    TensorOperation::prim_t::none,
                    TensorOperation::prim_t::gemm,
                    TensorOperation::prim_t::relu,
                    i_dim_types_span,
                    i_exec_types_span,
                    i_dim_sizes_span,
                    i_strides_in0_span,
                    i_strides_in1_span,
                    i_strides_out_span);

    tensor_op.compile();

    // create in0, in1 and out tensor
    int64_t size_in0 = 32 * 8 * 32 * 32;
    int64_t size_in1 = 32 * 8 * 32 * 32;
    int64_t size_out = 32 * 32 * 32 * 32;

    float* tensor_in0 = new float[size_in0];
    float* tensor_in1 = new float[size_in1];
    float* tensor_out = new float[size_out];
    float* tensor_out_ref = new float[size_out];

    // initialize input tensors
    srand(42);
    for (size_t i = 0; i < size_in0; i++) {
        tensor_in0[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < size_in1; i++) {
        tensor_in1[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < size_out; i++) {
        tensor_out[i] = 0.0f;
        tensor_out_ref[i] = 0.0f;
    }

    // execute reference
    example1_2_ref(tensor_in0, tensor_in1, tensor_out_ref);
    // apply ReLU to reference output
    for (size_t i = 0; i < size_out; i++) {
        tensor_out_ref[i] = std::max(0.0f, tensor_out_ref[i]);
    }
    // execute TenOp
    tensor_op.execute(tensor_in0, tensor_in1, tensor_out);

    // verify results
    double error = 0.0;
    for (size_t i = 0; i < size_out; i++) {
        error += std::abs(tensor_out[i] - tensor_out_ref[i]);
        if (std::abs(tensor_out[i] - tensor_out_ref[i]) > 1e-5) {
            std::cout << "Error at index " << i << ": " << tensor_out[i] << " vs " << tensor_out_ref[i] << std::endl;
            break;
        }
    }
    std::cout << "  Total error third example: " << error << std::endl;

    // benchmark execution time
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 200; i++) {
        tensor_op.execute(tensor_in0, tensor_in1, tensor_out);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "  Execution time for third example: " << elapsed.count() << " seconds" << std::endl;
    double gflops = tensor_op.get_flops_count() / elapsed.count() / 1e9;
    gflops *= 200;
    std::cout << "  GFLOPS for third example: " << gflops << std::endl;

    // cleanup
    delete[] tensor_in0;
    delete[] tensor_in1;
    delete[] tensor_out;
    delete[] tensor_out_ref;
}

int main() {
    std::cout << "Benchmarking Tensor contraction settings ..." << std::endl;

    first_example();
    std::cout << "----------------------------------------" << std::endl;
    second_example();
    std::cout << "----------------------------------------" << std::endl;
    third_example();

    return EXIT_SUCCESS;
}