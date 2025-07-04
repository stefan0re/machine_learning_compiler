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

TEST_CASE("MiniJit::Brgemm::FP32 test", "[MiniJit][GEMM][FP32]") {
    int k_sizes[5] = {1, 16, 32, 64, 128};
    for (size_t n = 1; n < 65;) {
        for (size_t m = 1; m < 65; m++) {
            mini_jit::generator::Brgemm l_brgemm;
            l_brgemm.generate(m, n, k, 1, 0, 0, 0, mini_jit::generator::Brgemm::dtype_t::fp32);

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
                l_c_ref[i] = l_c_1[i];
            }

            gemm_ref(l_a, l_b, l_c_ref,
                     m, n, k,
                     m, k, m);
            mini_jit::generator::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();
            l_kernel(l_a, l_b, l_c_jit, m, k, m, 0, 0);

            double l_error = 0.0;
            for (size_t i = 0; i < m * n; i++) {
                diff += std::abs(l_c_jit[i] - l_c_ref[i]);
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