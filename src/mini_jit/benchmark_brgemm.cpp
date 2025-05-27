#include <chrono>
#include <iostream>
#include <vector>

#include "generator/Brgemm.h"
#include "include/gemm_ref.h"

// test to check if brgemm works

int main(int argc, char *argv[]) {
    int64_t m = atoi(argv[1]);
    int64_t n = atoi(argv[2]);
    int64_t k = (argc > 3) ? atoi(argv[3]) : 1;
    int64_t br_k = (argc > 4) ? atoi(argv[4]) : 1;

    const int64_t lda = m;
    const int64_t ldb = k;
    const int64_t ldc = m;
    const int64_t br_stride_a = lda * k;
    const int64_t br_stride_b = ldb * n;

    std::cout << "Dimensions: M = " << m << ", N = " << n << ", K = " << k << ", BR_K = " << br_k << std::endl;

    mini_jit::generator::Brgemm l_brgemm;
    l_brgemm.generate(m, n, k, br_k, 0, 0, 0, mini_jit::generator::Brgemm::dtype_t::fp32);

    // initialize matrix
    float *l_a = (float *)malloc(lda * k * br_k * sizeof(float));
    float *l_b = (float *)malloc(ldb * n * br_k * sizeof(float));
    float *l_c_1 = (float *)malloc(ldc * n * sizeof(float));
    float *l_c_2 = (float *)malloc(ldc * n * sizeof(float));

    for (int i = 0; i < lda * k * br_k; i++) {
        l_a[i] = (float)drand48();
    }
    for (int i = 0; i < ldb * n * br_k; i++) {
        l_b[i] = (float)drand48();
    }
    for (int i = 0; i < ldc * n; i++) {
        l_c_1[i] = (float)drand48();
        l_c_2[i] = l_c_1[i];
    }

    // compute reference
    brgemm_ref(l_a, l_b, l_c_1,
               m, n, k, br_k,
               lda, ldb, ldc, br_stride_a, br_stride_b);

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

    if (l_diff < 1e-4) {
        l_diff = 0;
        std::cout << "Diff: " << l_diff << std::endl;
    } else {
        std::cout << "Diff: " << "ERROR" << std::endl;
    }
    std::cout << "===================" << std::endl;

    free(l_a);
    free(l_b);
    free(l_c_1);
    free(l_c_2);
}