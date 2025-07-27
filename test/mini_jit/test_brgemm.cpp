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

TEST_CASE("MiniJit::Brgemm::FP32 Tests BRGEMMs", "[MiniJit][GEMM][FP32]") {
    for (size_t l_i = 0; l_i < 4000; l_i++) {
        srand48(time(NULL));

        int64_t m = (int64_t)(drand48() * 64.0) + 1;
        int64_t n = (int64_t)(drand48() * 64.0) + 1;
        int64_t k = (int64_t)(drand48() * 64.0) + 1;
        int16_t br = (int64_t)(drand48() * 16.0) + 1;

        mini_jit::generator::Brgemm l_brgemm;
        l_brgemm.generate(m, n, k, br, 0, 0, 0, mini_jit::generator::Brgemm::dtype_t::fp32);

        float *l_a = (float *)malloc(m * k * br * sizeof(float));
        float *l_b = (float *)malloc(k * n * br * sizeof(float));
        float *l_c_jit = (float *)malloc(m * n * sizeof(float));
        float *l_c_ref = (float *)malloc(m * n * sizeof(float));

        for (int i = 0; i < br * m * k; i++) {
            l_a[i] = (float)drand48() * 10 - 5;
        }
        for (int i = 0; i < br * k * n; i++) {
            l_b[i] = (float)drand48() * 10 - 5;
        }
        for (int i = 0; i < m * n; i++) {
            l_c_jit[i] = (float)drand48() * 10 - 5;
            l_c_ref[i] = l_c_jit[i];
        }

        brgemm_ref(l_a, l_b, l_c_ref,
                   m, n, k, br,
                   m, k, m,
                   m * k, n * k);
        mini_jit::generator::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();
        l_kernel(l_a, l_b, l_c_jit, m, k, m, m * k, n * k);

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