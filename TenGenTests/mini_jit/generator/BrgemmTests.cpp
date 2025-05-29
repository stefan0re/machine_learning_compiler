#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "TenGen.h"
#include "TenGenTests/TenGenTestsHelper.h"

using Brgemm = TenGen::MiniJit::Generator::Brgemm;

TEST_CASE("MiniJit::Brgemm::FP32 test", "[MiniJit][Brgemm][FP32]") {
    int64_t m = GENERATE(8, 16);
    int64_t n = GENERATE(8, 16);
    int64_t k = GENERATE(1, 2);
    int64_t br_k = GENERATE(1, 2);

    const int64_t lda = m;
    const int64_t ldb = k;
    const int64_t ldc = m;
    const int64_t br_stride_a = lda * k;
    const int64_t br_stride_b = ldb * n;

    Brgemm l_brgemm;
    l_brgemm.generate(m, n, k, br_k, 0, 0, 0, dtype_t::fp32);

    float* l_a = (float*)malloc(lda * k * br_k * sizeof(float));
    float* l_b = (float*)malloc(ldb * n * br_k * sizeof(float));
    float* l_c_1 = (float*)malloc(ldc * n * sizeof(float));
    float* l_c_2 = (float*)malloc(ldc * n * sizeof(float));

    REQUIRE(l_a != nullptr);
    REQUIRE(l_b != nullptr);
    REQUIRE(l_c_1 != nullptr);
    REQUIRE(l_c_2 != nullptr);

    for (int i = 0; i < lda * k * br_k; i++) {
        l_a[i] = static_cast<float>(drand48());
    }
    for (int i = 0; i < ldb * n * br_k; i++) {
        l_b[i] = static_cast<float>(drand48());
    }
    for (int i = 0; i < ldc * n; i++) {
        l_c_1[i] = static_cast<float>(drand48());
        l_c_2[i] = l_c_1[i];
    }

    // reference result
    TenGenTestsHelper::brgemm_ref(
        l_a, l_b, l_c_1, m, n, k, br_k,
        lda, ldb, ldc, br_stride_a, br_stride_b);

    // jitted result
    auto kernel = l_brgemm.get_kernel();
    kernel(l_a, l_b, l_c_2, lda, ldb, ldc, br_stride_a, br_stride_b);

    double diff = 0.0;
    for (int i = 0; i < ldc * n; i++) {
        diff += std::abs(l_c_1[i] - l_c_2[i]);
        REQUIRE(std::abs(l_c_1[i] - l_c_2[i]) < 0.0001);
    }

    INFO("M = " << m << ", N = " << n << ", K = " << k << ", BR_K = " << br_k);
    INFO("Diff: " << diff);

    free(l_a);
    free(l_b);
    free(l_c_1);
    free(l_c_2);
}
