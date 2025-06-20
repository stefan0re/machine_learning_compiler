#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../../src/mini_jit/generator/Brgemm.h"
#include "../../src/mini_jit/generator/Unary.h"
#include "../../src/mini_jit/generator/Util.h"
#include "../../src/mini_jit/include/gemm_ref.h"

using namespace mini_jit::generator;

void generate_matrix(uint32_t height, uint32_t width, float* M, bool set_zero = false, bool visualization = false) {
    float MAX = 100;
    float MIN = -100;
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int index = j * height + i;
            float rand_float = static_cast<float>(rand()) / RAND_MAX;
            rand_float = rand_float * (MAX - MIN) + MIN;
            if (visualization) {
                rand_float = std::trunc(rand_float);
            }
            M[index] = (1 - (double)set_zero) * rand_float;
        }
    }
}

bool compare_matrix(uint32_t height, uint32_t width, float* M, float* C) {
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int index = j * height + i;
            if (M[index] != C[index]) {
                std::cout << "Matrices are not equal in element: " << index << " M: " << M[index] << ", C: " << C[index] << std::endl;
                return false;
            }
        }
    }
    return true;
}

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

    // Generate the BRGEMM kernel

    Brgemm l_brgemm;
    l_brgemm.generate(m, n, k, br_k, 0, 0, 0, Brgemm::dtype_t::fp32);

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
    brgemm_ref(l_a, l_b, l_c_1,
               m, n, k, br_k,
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

TEST_CASE("Unary generate zero kernel sets all elements to zero", "[unary][generate_zero]") {
    Util::KernelSize kernelSize;
    // this works only for kernels like 4 x 4
    // THIS IS NOT WORKING:
    // kernelSize.M = 37;
    // kernelSize.N = 114;

    kernelSize.M = 32;
    kernelSize.N = 32;

    size_t size = kernelSize.M * kernelSize.N;
    int leading_dimension = kernelSize.M;

    float* a = new float[size];
    float* b = new float[size];
    float* c = new float[size];

    Unary unary;

    generate_matrix(kernelSize.M, kernelSize.N, a);
    generate_matrix(kernelSize.M, kernelSize.N, b);
    generate_matrix(kernelSize.M, kernelSize.N, c, true);  // fill with zeroes

    unary.generate(kernelSize.M, kernelSize.N, Unary::dtype_t::fp32, Unary::ptype_t::zero);

    Unary::kernel_t zero = unary.get_kernel();

    zero(a, b, leading_dimension, leading_dimension);

    bool match = compare_matrix(kernelSize.M, kernelSize.N, b, c);
    REQUIRE(match == true);

    delete[] a;
    delete[] b;
    delete[] c;
}
