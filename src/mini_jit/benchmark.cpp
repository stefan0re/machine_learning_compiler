#include <chrono>
#include <iostream>
#include <vector>

#include "generator/Brgemm.h"
#include "include/gemm_ref.h"

/* currently just testing if load and store C works (copy) */

int main(int argc, char *argv[]) {
    // std::cout << "mini_jit benchmark" << std::endl;
    // std::cout << "===================" << std::endl;

    int64_t m = atoi(argv[1]);
    int64_t n = atoi(argv[2]);
    int64_t k = (argc > 3) ? atoi(argv[3]) : 1;

    const int64_t lda = m;
    const int64_t ldb = k;
    const int64_t ldc = m;
    const int64_t br_stride_a = m * k;
    const int64_t br_stride_b = k * n;

    std::cout << "Dimensions: M = " << m << ", N = " << n << ", K = " << k << std::endl;
    // std::cout << "Leading dims: A = " << lda << ", B = " << ldb << ", C = " << ldc << std::endl;
    // std::cout << "Brgemm stride: A = " << br_stride_a << ", B = " << br_stride_b << std::endl;

    mini_jit::generator::Brgemm l_brgemm;
    l_brgemm.generate(m, n, k, 1, 0, 0, 0, mini_jit::generator::Brgemm::dtype_t::fp32);

    // initialize matrix
    float *l_a = (float *)malloc(lda * k * sizeof(float));
    float *l_b = (float *)malloc(ldb * n * sizeof(float));
    float *l_c_1 = (float *)malloc(ldc * n * sizeof(float));
    float *l_c_2 = (float *)malloc(ldc * n * sizeof(float));

    for (int i = 0; i < lda * k; i++) {
        l_a[i] = (float)drand48();
    }
    for (int i = 0; i < ldb * n; i++) {
        l_b[i] = (float)drand48();
    }
    for (int i = 0; i < ldc * n; i++) {
        l_c_1[i] = (float)drand48();
        l_c_2[i] = l_c_1[i];
    }

    // compute reference
    gemm_ref(l_a, l_b, l_c_1,
             m, n, k,
             lda, ldb, ldc);

    // compute jiter
    mini_jit::generator::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();

    l_kernel(l_a, l_b, l_c_2,
             lda, ldb, ldc,
             br_stride_a, br_stride_b);

    // compare results
    double l_diff = 0.0;
    for (int i = 0; i < ldc * n; i++) {
        l_diff += fabs(l_c_1[i] - l_c_2[i]);
        if (fabs(l_c_1[i] - l_c_2[i]) > 0.0001) {
            // std::cout << "Error: " << l_c_1[i] << " != " << l_c_2[i] << std::endl;
        }
    }

    if (l_diff < 1e-4) l_diff = 0;

    std::cout << "Diff: " << l_diff << std::endl;
    std::cout << "===================" << std::endl;

    free(l_a);
    free(l_b);
    free(l_c_1);
    free(l_c_2);
}