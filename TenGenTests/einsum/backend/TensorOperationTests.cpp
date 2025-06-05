#include <TenGen.h>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>

#include "../../TenGenTestsHelper.h"

static constexpr size_t SIZE_TENSOR_IN = 32 * 8 * 32 * 32;
static constexpr size_t SIZE_TENSOR_OUT = 32 * 32 * 32 * 32;

static void init_random(float* tensor, size_t size) {
    srand48(0);
    for (size_t i = 0; i < size; ++i) {
        tensor[i] = (10 * static_cast<float>(drand48())) - 5.0f;
    }
}

static void zero_tensor(float* tensor, size_t size) {
    std::fill(tensor, tensor + size, 0.0f);
}

TEST_CASE("Example 1: Scalar vs Einsum without br_gemm and no relu", "[einsum]") {
    float* l_ten_1 = new float[SIZE_TENSOR_IN];
    float* l_ten_2 = new float[SIZE_TENSOR_IN];
    float* l_out_scalar = new float[SIZE_TENSOR_OUT];
    float* l_out_einsum_1 = new float[SIZE_TENSOR_OUT];

    init_random(l_ten_1, SIZE_TENSOR_IN);
    init_random(l_ten_2, SIZE_TENSOR_IN);
    zero_tensor(l_out_scalar, SIZE_TENSOR_OUT);
    zero_tensor(l_out_einsum_1, SIZE_TENSOR_OUT);

    TenGenTestsHelper::run_1_example_with_scalar(l_ten_1, l_ten_2, l_out_scalar);
    TenGenTestsHelper::run_example_with_einsum(l_ten_1, l_ten_2, l_out_einsum_1, false, false, false);

    bool equal = TenGenTestsHelper::check_diff(l_out_scalar, l_out_einsum_1, SIZE_TENSOR_OUT);
    REQUIRE(equal);

    delete[] l_ten_1;
    delete[] l_ten_2;
    delete[] l_out_scalar;
    delete[] l_out_einsum_1;
}

TEST_CASE("Example 2: Scalar vs Einsum with br_gemm, no relu", "[einsum]") {
    float* l_ten_1 = new float[SIZE_TENSOR_IN];
    float* l_ten_2 = new float[SIZE_TENSOR_IN];
    float* l_out_scalar = new float[SIZE_TENSOR_OUT];
    float* l_out_einsum_2 = new float[SIZE_TENSOR_OUT];

    init_random(l_ten_1, SIZE_TENSOR_IN);
    init_random(l_ten_2, SIZE_TENSOR_IN);
    zero_tensor(l_out_scalar, SIZE_TENSOR_OUT);
    zero_tensor(l_out_einsum_2, SIZE_TENSOR_OUT);

    TenGenTestsHelper::run_1_example_with_scalar(l_ten_1, l_ten_2, l_out_scalar);
    TenGenTestsHelper::run_example_with_einsum(l_ten_1, l_ten_2, l_out_einsum_2, true, false, false);

    bool equal = TenGenTestsHelper::check_diff(l_out_scalar, l_out_einsum_2, SIZE_TENSOR_OUT);
    REQUIRE(equal);

    delete[] l_ten_1;
    delete[] l_ten_2;
    delete[] l_out_scalar;
    delete[] l_out_einsum_2;
}

TEST_CASE("Example 3: Scalar vs Einsum with br_gemm, relu, and first_touch_zero", "[einsum]") {
    float* l_ten_1 = new float[SIZE_TENSOR_IN];
    float* l_ten_2 = new float[SIZE_TENSOR_IN];
    float* l_out_scalar = new float[SIZE_TENSOR_OUT];
    float* l_out_einsum_3 = new float[SIZE_TENSOR_OUT];

    init_random(l_ten_1, SIZE_TENSOR_IN);
    init_random(l_ten_2, SIZE_TENSOR_IN);
    zero_tensor(l_out_scalar, SIZE_TENSOR_OUT);
    zero_tensor(l_out_einsum_3, SIZE_TENSOR_OUT);

    TenGenTestsHelper::run_1_example_with_scalar(l_ten_1, l_ten_2, l_out_scalar);
    TenGenTestsHelper::run_example_with_einsum(l_ten_1, l_ten_2, l_out_einsum_3, true, true, true);

    for (size_t i = 0; i < SIZE_TENSOR_OUT; ++i) {
        l_out_scalar[i] = std::max(0.0f, l_out_scalar[i]);
    }

    bool equal = TenGenTestsHelper::check_diff(l_out_scalar, l_out_einsum_3, SIZE_TENSOR_OUT);
    REQUIRE(equal);

    delete[] l_ten_1;
    delete[] l_ten_2;
    delete[] l_out_scalar;
    delete[] l_out_einsum_3;
}
