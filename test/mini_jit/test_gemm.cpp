#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../../src/mini_jit/generator/Brgemm.h"
#include "../../src/mini_jit/generator/Unary.h"
#include "../../src/mini_jit/generator/Util.h"
#include "../../src/mini_jit/include/gemm_ref.h"
#include "../test_utils/test_utils.h"

using namespace mini_jit::generator;

TEST_CASE("MiniJit::Brgemm::FP32 Test all GEMMs", "[MiniJit][GEMM][FP32]") {
    int k_sizes[5] = {1, 16, 32, 64, 128};
    for (size_t n = 1; n < 65; n++) {
        for (size_t m = 1; m < 65; m++) {
            for (int k : k_sizes) {
                mini_jit::generator::Brgemm l_brgemm;
                l_brgemm.generate(m, n, k, 1, 0, 0, 0, mini_jit::generator::Brgemm::dtype_t::fp32, false);

                float *l_a = (float *)malloc(m * k * sizeof(float));
                float *l_b = (float *)malloc(k * n * sizeof(float));
                float *l_c_jit = (float *)malloc(m * n * sizeof(float));
                float *l_c_ref = (float *)malloc(m * n * sizeof(float));

                srand48(time(NULL));

                for (int i = 0; i < m * k; i++) {
                    l_a[i] = (float)drand48() * 10 - 5;
                }
                for (int i = 0; i < k * n; i++) {
                    l_b[i] = (float)drand48() * 10 - 5;
                }
                for (int i = 0; i < m * n; i++) {
                    l_c_jit[i] = (float)drand48() * 10 - 5;
                    l_c_ref[i] = l_c_jit[i];
                }

                gemm_ref(l_a, l_b, l_c_ref,
                         m, n, k,
                         m, k, m);
                mini_jit::generator::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();
                l_kernel(l_a, l_b, l_c_jit, m, k, m, 0, 0);

                double l_error = 0.0;
                for (size_t i = 0; i < m * n; i++) {
                    l_error += std::abs(l_c_jit[i] - l_c_ref[i]);
                    REQUIRE(std::abs(l_c_jit[i] - l_c_ref[i]) < 0.0001);
                }
                REQUIRE(l_error < 1e-3);
                free(l_a);
                free(l_b);
                free(l_c_jit);
                free(l_c_ref);
            }
        }
    }
}

TEST_CASE("MiniJit::Brgemm::FP32 Test bigger LD", "[MiniJit][GEMM][FP32]") {
    int k_sizes[4] = {1, 16, 32, 64};
    for (size_t n = 1; n < 65; n++) {
        for (size_t m = 1; m < 65; m++) {
            for (int k : k_sizes) {
                mini_jit::generator::Brgemm l_brgemm;
                l_brgemm.generate(m, n, k, 1, 0, 0, 0, mini_jit::generator::Brgemm::dtype_t::fp32, false);

                int l_lda = (int)(drand48() * 100) + 64;
                int l_ldb = (int)(drand48() * 100) + 64;
                int l_ldc = (int)(drand48() * 100) + 64;

                float *l_a = (float *)malloc(l_lda * k * sizeof(float));
                float *l_b = (float *)malloc(l_ldb * n * sizeof(float));
                float *l_c_jit = (float *)malloc(l_ldc * n * sizeof(float));
                float *l_c_ref = (float *)malloc(l_ldc * n * sizeof(float));

                srand48(time(NULL));

                for (int i = 0; i < l_lda * k; i++) {
                    l_a[i] = (float)drand48() * 10 - 5;
                }
                for (int i = 0; i < l_ldb * n; i++) {
                    l_b[i] = (float)drand48() * 10 - 5;
                }
                for (int i = 0; i < l_ldc * n; i++) {
                    l_c_jit[i] = (float)drand48() * 10 - 5;
                    l_c_ref[i] = l_c_jit[i];
                }

                gemm_ref(l_a, l_b, l_c_ref,
                         m, n, k,
                         l_lda, l_ldb, l_ldc);
                mini_jit::generator::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();
                l_kernel(l_a, l_b, l_c_jit, l_lda, l_ldb, l_ldc, 0, 0);

                double l_error = 0.0;
                for (size_t i = 0; i < m * n; i++) {
                    l_error += std::abs(l_c_jit[i] - l_c_ref[i]);
                    REQUIRE(std::abs(l_c_jit[i] - l_c_ref[i]) < 0.0001);
                }
                REQUIRE(l_error < 1e-4);
                free(l_a);
                free(l_b);
                free(l_c_jit);
                free(l_c_ref);
            }
        }
    }
}