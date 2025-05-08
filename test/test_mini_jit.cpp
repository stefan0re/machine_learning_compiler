#include <time.h>

#include "../src/mini_jit/generator/Brgemm.h"
#include "../src/mini_jit/include/gemm_ref.h"

int test_gemm_16_6_k(int64_t i_k,
                     int64_t i_lda,
                     int64_t i_ldb,
                     int64_t i_ldc) {
    mini_jit::generator::Brgemm l_brgemm;
    l_brgemm.generate(16, 6, i_k, 1, 0, 0, 0, mini_jit::generator::Brgemm::dtype_t::fp32);

    srand48(time(NULL));

    // generate random A B and C
    srand48(time(NULL));

    // initialize matrix
    float *l_a = (float *)malloc(i_lda * i_k * sizeof(float));
    float *l_b = (float *)malloc(i_ldb * 6 * sizeof(float));
    float *l_c_1 = (float *)malloc(i_ldc * 6 * sizeof(float));
    float *l_c_2 = (float *)malloc(i_ldc * 6 * sizeof(float));

    for (int i = 0; i < i_lda * i_k; i++) {
        l_a[i] = (float)drand48();
    }
    for (int i = 0; i < i_ldb * 6; i++) {
        l_b[i] = (float)drand48();
    }
    for (int i = 0; i < i_ldc * 6; i++) {
        l_c_1[i] = (float)drand48();
        l_c_2[i] = l_c_1[i];
    }

    // compute reference
    gemm_ref(l_a, l_b, l_c_1,
             16, 6, i_k,
             i_lda, i_ldb, i_ldb);

    // compute jiter
    mini_jit::generator::Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();

    l_kernel(l_a, l_b, l_c_2,
             i_lda, i_ldb, i_ldb,
             0, 0);

    // compare results
    double l_diff = 0.0;
    for (int i = 0; i < i_ldc * 6; i++) {
        l_diff += fabs(l_c_1[i] - l_c_2[i]);
    }
    if (l_diff > 0.0005) {
        return -1;
    } else {
        return 0;
    }

    free(l_a);
    free(l_b);
    free(l_c_1);
    free(l_c_2);
}

int main() {
    srand48(time(NULL));

    // create random ints from 100 to 200
    int64_t l_lda = 128 + (int64_t)(drand48() * 128);
    int64_t l_ldb = 128 + (int64_t)(drand48() * 128);
    int64_t l_ldc = 128 + (int64_t)(drand48() * 128);

    int l_error = 0;

    for (size_t k = 1; k < 128; k++) {
        l_error += test_gemm_16_6_k(k, l_lda, l_ldb, l_ldc);
    }

    return l_error;
}