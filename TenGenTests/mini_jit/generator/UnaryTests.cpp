#include <catch2/catch_test_macros.hpp>

#include "../../TenGenTestsHelper.h"
#include "TenGen.h"

using Unary = TenGen::MiniJit::Generator::Unary;
using namespace TenGen::Structs;
using namespace TenGen::Types;

TEST_CASE("Unary generate zero kernel sets all elements to zero", "[unary][generate_zero]") {
    KernelSize kernelSize;
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

    TenGenTestsHelper::generate_matrix(kernelSize.M, kernelSize.N, a);
    TenGenTestsHelper::generate_matrix(kernelSize.M, kernelSize.N, b);
    TenGenTestsHelper::generate_matrix(kernelSize.M, kernelSize.N, c, true);  // fill with zeroes

    unary.generate(kernelSize.M, kernelSize.N, 0, dtype_t::fp32, ptype_t::zero);

    Unary::kernel_t zero = unary.get_kernel();

    zero(a, b, leading_dimension, leading_dimension);

    bool match = TenGenTestsHelper::compare_matrix(kernelSize.M, kernelSize.N, b, c);
    REQUIRE(match == true);

    delete[] a;
    delete[] b;
    delete[] c;
}
