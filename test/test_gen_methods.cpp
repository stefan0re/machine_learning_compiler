#include <math.h>

#include <iostream>

#include "../src/mini_jit/generator/Unary.h"
#include "../src/mini_jit/generator/Util.h"
#include "./test_utils.h"

using namespace mini_jit::generator;

int test_get_kernel_sizes() {
    Util::KernelSize kernelSize;
    kernelSize.M = 114;
    kernelSize.N = 74;

    Util::KernelSizes kernelSizes = Util::KernelSizes{Util::KernelSize{0, 0},
                                                      Util::KernelSize{0, 0},
                                                      Util::KernelSize{0, 0},
                                                      Util::KernelSize{0, 0}};
    Util::KernelSizes ref_kernelSizes = Util::KernelSizes{Util::KernelSize{16, 4},
                                                          Util::KernelSize{16, 5},
                                                          Util::KernelSize{10, 8},
                                                          Util::KernelSize{10, 10}};

    Util::get_kernel_sizes(kernelSize.M, kernelSize.N, kernelSizes, false);

    bool match = kernelSizes.kernel1.M == ref_kernelSizes.kernel1.M && kernelSizes.kernel1.N == ref_kernelSizes.kernel1.N;
    return match ? 0 : -1;
}

int test_generate_zero() {
    Util::KernelSize kernelSize;
    kernelSize.M = 37;
    kernelSize.N = 114;
    size_t size = kernelSize.M * kernelSize.N;
    int leading_dimension = kernelSize.M;

    float* a = new float[size];
    float* b = new float[size];
    Unary unary;

    test_utils::generate_matrix(kernelSize.M, kernelSize.N, a);
    test_utils::generate_matrix(kernelSize.M, kernelSize.N, b);

    unary.generate(kernelSize.M, kernelSize.N, 0, Unary::dtype_t::fp32, Unary::ptype_t::zero);

    Unary::kernel_t zero = unary.get_kernel();

    zero(a, b, leading_dimension, leading_dimension);

    float* c = new float[size];
    test_utils::generate_matrix(kernelSize.M, kernelSize.N, c, true);
    bool match = test_utils::compare_matrix(kernelSize.M, kernelSize.N, b, c);

    delete[] a;
    delete[] b;
    delete[] c;

    return match ? 0 : -1;
}

int test_generate_relu() {
    Util::KernelSize kernelSize;
    kernelSize.M = 2048;
    kernelSize.N = 2048;
    size_t size = kernelSize.M * kernelSize.N;
    int leading_dimension = kernelSize.M;

    float* a = new float[size];
    float* b = new float[size];

    Unary unary;

    test_utils::generate_matrix(kernelSize.M, kernelSize.N, a);
    test_utils::generate_matrix(kernelSize.M, kernelSize.N, b);
    unary.generate(kernelSize.M, kernelSize.N, 0, Unary::dtype_t::fp32, Unary::ptype_t::relu);
    Unary::kernel_t relu = unary.get_kernel();

    relu(a, b, leading_dimension, leading_dimension);
    float* c = new float[size];

    for (int i = 0; i < kernelSize.M; i++) {
        for (int j = 0; j < kernelSize.N; j++) {
            int index = j * kernelSize.M + i;
            c[index] = std::max(0.f, a[index]);
        }
    }

    bool match = test_utils::compare_matrix(kernelSize.M, kernelSize.N, b, c);

    delete[] a;
    delete[] b;
    delete[] c;

    return match ? 0 : -1;
}

int test_generate_identity() {
    Util::KernelSize kernelSize;
    kernelSize.M = 4;
    kernelSize.N = 4;
    int leading_dimension = kernelSize.M;

    float a[kernelSize.M * kernelSize.N];
    float b[kernelSize.M * kernelSize.N];
    Unary unary;

    test_utils::generate_matrix(kernelSize.M, kernelSize.N, a, false, true);
    test_utils::generate_matrix(kernelSize.M, kernelSize.N, b, true);

    unary.generate(kernelSize.M, kernelSize.N, 1, Unary::dtype_t::fp32, Unary::ptype_t::identity);

    Unary::kernel_t identity = unary.get_kernel();

    identity(a, b, leading_dimension, leading_dimension);

    float c[kernelSize.M * kernelSize.N];
    test_utils::generate_matrix(kernelSize.M, kernelSize.N, c, true);
    test_utils::transpose_matrix(kernelSize.M, kernelSize.N, a, c);
    bool match = test_utils::compare_matrix(kernelSize.M, kernelSize.N, c, b);

    // test_utils::visualize_matrix(kernelSize.M, kernelSize.N, a, "A");
    // test_utils::visualize_matrix(kernelSize.M, kernelSize.N, b, "B");
    // test_utils::visualize_matrix(kernelSize.M, kernelSize.N, c, "C");

    return match ? 0 : -1;
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    int result = 0;

    result |= test_get_kernel_sizes();
    result |= test_generate_zero();
    result |= test_generate_relu();
    result |= test_generate_identity();

    return result;
}