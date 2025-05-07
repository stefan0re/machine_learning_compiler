#include <iostream>
#include <vector>

#include "generator/Brgemm.h"

int main() {
    std::cout << "Running mini_jit ..." << std::endl;

    int64_t m = 2;
    int64_t n = 2;
    int64_t k = 64;

    mini_jit::generator::Brgemm l_brgemm;
    l_brgemm.generate(m, n, k, 1, 0, 0, 0, mini_jit::generator::Brgemm::dtype_t::fp32);

    std::vector<float> A(m * k, 1.f);
    std::vector<float> B(k * n, 1.f);
    std::vector<float> C(m * n, 0.f);

    const int64_t lda = m;
    const int64_t ldb = k;
    const int64_t ldc = m;
    const int64_t br_stride_a = m * k;
    const int64_t br_stride_b = k * n;

    mini_jit::generator::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();

    l_kernel(A.data(), B.data(), C.data(),
             lda, ldb, ldc,
             br_stride_a, br_stride_b);
}