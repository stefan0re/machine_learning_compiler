#include <chrono>
#include <iostream>
#include <span>

#include "../mini_jit/generator/Brgemm.h"
#include "backend/TensorOperation.h"

/**
 * Example 1 (einsum expression) abdc, ebfd -> aefc
    - dtype	FP32
    - prim_first_touch	None
    - prim_main	GEMM
    - prim_last_touch	None
    - dim_types  ( M, N, K, M, N, K )
    - exec_types ( Seq, Seq, Seq, Prim, Prim, Prim )
    - dim_sizes	 ( 32, 32, 8, 32, 32, 32 )
    - strides_in0	( 8192, 0, 1024, 1, 0, 32 )
    - strides_in1	( 0, 8192, 1024, 0, 32, 1 )
    - strides_out	( 32768, 1024, 0, 1, 32, 0 )

    aufbau input1: 32x8x32x32 -> abdc
    aufbau input2: 32x8x32x32 -> ebfd
    aufbau output: 32x32x32x32 -> aefc


                      l_oM   l_oK    l_iK  l_iM
    strides input1:  8192 -> 1024 -> 32 -> 1

                     l_oN    l_oK    l_iN  l_iK
    strides input2:  8192 -> 1024 -> 32 -> 1

                     l_oM    l_oN    l_iN  l_iM
    strides output: 32768 -> 1024 -> 32 -> 1
*/

using namespace einsum::backend;

void run_1_example_with_scalar(float* i_ten_1,
                               float* i_ten_2,
                               float* o_ten) {
    for (size_t l_oM = 0; l_oM < 32; ++l_oM) {
        for (size_t l_oN = 0; l_oN < 32; ++l_oN) {
            for (size_t l_oK = 0; l_oK < 8; ++l_oK) {
                for (size_t l_iM = 0; l_iM < 32; ++l_iM) {
                    for (size_t l_iN = 0; l_iN < 32; ++l_iN) {
                        for (size_t l_iK = 0; l_iK < 32; ++l_iK) {
                            size_t l_idx_1 = l_oM * 8192 + l_oK * 1024 + l_iK * 32 + l_iM;
                            size_t l_idx_2 = l_oN * 8192 + l_oK * 1024 + l_iN * 32 + l_iK;
                            size_t l_idx_out = l_oM * 32768 + l_oN * 1024 + l_iN * 32 + l_iM;

                            o_ten[l_idx_out] += i_ten_1[l_idx_1] * i_ten_2[l_idx_2];
                        }
                    }
                }
            }
        }
    }
}

void run_1_example_with_gemm(float* i_ten_1,
                             float* i_ten_2,
                             float* o_ten) {
    mini_jit::generator::Brgemm l_brgemm;
    l_brgemm.generate(32, 32, 32, 1, 0, 0, 0, mini_jit::generator::Brgemm::dtype_t::fp32);
    mini_jit::generator::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();

    for (size_t l_oM = 0; l_oM < 32; ++l_oM) {
        for (size_t l_oN = 0; l_oN < 32; ++l_oN) {
            for (size_t l_oK = 0; l_oK < 8; ++l_oK) {
                float* l_a = &i_ten_1[l_oM * 8192 + l_oK * 1024];
                float* l_b = &i_ten_2[l_oN * 8192 + l_oK * 1024];
                float* l_c = &o_ten[l_oM * 32768 + l_oN * 1024];
                l_kernel(l_a, l_b, l_c, 32, 32, 32, 8192, 8192);
            }
        }
    }
}

void run_1_example_with_einsum(float* i_ten_1,
                               float* i_ten_2,
                               float* o_ten,
                               bool i_relu,
                               bool i_first_touch_zero) {
    TensorOperation l_tensor_op;

    TensorOperation::dtype_t l_dtype = TensorOperation::dtype_t::fp32;
    TensorOperation::prim_t l_prim_first_touch = TensorOperation::prim_t::none;
    TensorOperation::prim_t l_prim_main = TensorOperation::prim_t::gemm;
    TensorOperation::prim_t l_prim_last_touch = TensorOperation::prim_t::none;

    std::vector<TensorOperation::dim_t> l_dim_types = {TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k,
                                                       TensorOperation::dim_t::m,
                                                       TensorOperation::dim_t::n,
                                                       TensorOperation::dim_t::k};

    std::vector<TensorOperation::exec_t> l_exec_types = {TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::seq,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim,
                                                         TensorOperation::exec_t::prim};

    std::vector<int64_t> l_dim_sizes = {32, 32, 8, 32, 32, 32};

    std::vector<int64_t> l_strides_in0 = {8192, 0, 1024, 1, 0, 32};
    std::vector<int64_t> l_strides_in1 = {0, 8192, 1024, 0, 32, 1};
    std::vector<int64_t> l_strides_out = {32768, 1024, 0, 1, 32, 0};

    // Setup the tensor operation
    auto l_error = l_tensor_op.setup(l_dtype,
                                     l_prim_first_touch,
                                     l_prim_main,
                                     l_prim_last_touch,
                                     std::span<const TensorOperation::dim_t>(l_dim_types),
                                     std::span<const TensorOperation::exec_t>(l_exec_types),
                                     std::span<const int64_t>(l_dim_sizes),
                                     std::span<const int64_t>(l_strides_in0),
                                     std::span<const int64_t>(l_strides_in1),
                                     std::span<const int64_t>(l_strides_out));

    l_tensor_op.execute(i_ten_1, i_ten_2, o_ten);
}

int main() {
    std::cout << "Benchmarking a few einsum Expressions..." << std::endl;

    // Initialize tensors
    float* l_ten_1 = new float[32 * 8 * 32 * 32];
    float* l_ten_2 = new float[32 * 8 * 32 * 32];

    float* l_out_scalar = new float[32 * 32 * 32 * 32];
    float* l_out_gemm = new float[32 * 32 * 32 * 32];
    float* l_out_einsum = new float[32 * 32 * 32 * 32];

    srand48(0);
    for (size_t i = 0; i < 32 * 8 * 32 * 32; ++i) {
        l_ten_1[i] = static_cast<float>(drand48());
    }
    for (size_t i = 0; i < 32 * 8 * 32 * 32; ++i) {
        l_ten_2[i] = static_cast<float>(drand48());
    }
    // Initialize output tensors to zero
    for (size_t i = 0; i < 32 * 32 * 32 * 32; ++i) {
        l_out_scalar[i] = 0.0f;
        l_out_gemm[i] = 0.0f;
        l_out_einsum[i] = 0.0f;
    }

    // run both implementations
    run_1_example_with_scalar(l_ten_1, l_ten_2, l_out_scalar);
    run_1_example_with_gemm(l_ten_1, l_ten_2, l_out_gemm);
    run_1_example_with_einsum(l_ten_1, l_ten_2, l_out_einsum, false, false);

    std::cout << "Finished tests" << std::endl;

    // Check if the results are equal
    bool equal = true;
    double l_max_diff = 0.0;
    for (size_t i = 0; i < 32 * 32 * 32 * 32; ++i) {
        if (std::abs(l_out_scalar[i] - l_out_einsum[i]) > 1e-3f) {
            equal = false;
            // std::cout << "i: " << i << ", scalar: " << l_out_scalar[i] << ", einsum: " << l_out_einsum[i] << std::endl;
            l_max_diff = std::max(l_max_diff, static_cast<double>(std::abs(l_out_scalar[i] - l_out_einsum[i])));
        }
    }

    if (equal) {
        std::cout << "Results are equal!" << std::endl;
    } else {
        std::cout << "Results are NOT equal!" << std::endl;
    }
    std::cout << "Max difference: " << l_max_diff << std::endl;

    // clean up
    delete[] l_ten_1;
    delete[] l_ten_2;
    delete[] l_out_scalar;
    delete[] l_out_gemm;
    delete[] l_out_einsum;

    std::cout << "Finished benchmarking." << std::endl;

    return EXIT_SUCCESS;
}