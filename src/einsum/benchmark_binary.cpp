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

void run_1_example_scalar(float* i_ten_1,
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

int main() {
    std::cout << "Benchmarking a few einsum Expressions..." << std::endl;

    // Initialize tensors
    float* l_ten_1 = new float[32 * 8 * 32 * 32];
    float* l_ten_2 = new float[32 * 8 * 32 * 32];

    float* l_out_scalar = new float[32 * 32 * 32 * 32];
    float* l_out_gemm = new float[32 * 32 * 32 * 32];

    srand48(0);  // Seed the random number generator
    // Fill tensors with random values
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
    }

    // run both implementations
    run_1_example_scalar(l_ten_1, l_ten_2, l_out_scalar);
    run_1_example_with_gemm(l_ten_1, l_ten_2, l_out_gemm);

    // Check if the results are equal
    bool equal = true;
    for (size_t i = 0; i < 32 * 32 * 32 * 32; ++i) {
        if (std::abs(l_out_scalar[i] - l_out_gemm[i]) > 1e-3) {
            equal = false;
            std::cout << "i: " << i << ", scalar: " << l_out_scalar[i] << ", gemm: " << l_out_gemm[i] << std::endl;
            break;
        }
    }
    if (equal) {
        std::cout << "Results are equal!" << std::endl;
    } else {
        std::cout << "Results are NOT equal!" << std::endl;
    }

    // clean up
    delete[] l_ten_1;
    delete[] l_ten_2;
    delete[] l_out_scalar;
    delete[] l_out_gemm;

    std::cout << "Finished benchmarking." << std::endl;

    return EXIT_SUCCESS;
}