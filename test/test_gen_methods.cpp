#include <iostream>

#include "../src/mini_jit/generator/Util.h"

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

    Util::get_kernel_sizes(kernelSize.M, kernelSize.N, kernelSizes);

    bool match = kernelSizes.kernel1.M == ref_kernelSizes.kernel1.M && kernelSizes.kernel1.N == ref_kernelSizes.kernel1.N;
    return match ? 0 : -1;
}

int main() {
    int result = 0;

    result |= test_get_kernel_sizes();

    return result;
}