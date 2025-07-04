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

    test::matmul::generate_matrix(kernelSize.M, kernelSize.N, a);
    test::matmul::generate_matrix(kernelSize.M, kernelSize.N, b);
    test::matmul::generate_matrix(kernelSize.M, kernelSize.N, c, true);  // fill with zeroes

    unary.generate(kernelSize.M, kernelSize.N, Unary::dtype_t::fp32, Unary::ptype_t::zero);

    Unary::kernel_t zero = unary.get_kernel();

    zero(a, b, leading_dimension, leading_dimension);

    bool match = test::matmul::compare_matrix(kernelSize.M, kernelSize.N, b, c);
    REQUIRE(match == true);

    delete[] a;
    delete[] b;
    delete[] c;
}
