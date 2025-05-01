#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>

extern "C" {
/**
 * @param a pointer to column-major matrix A.
 * @param b pointer to column-major matrix B.
 * @param c pointer to column-major matrix C.
 * @param lda leading dimension of A.
 * @param ldb leading dimension of B.
 * @param ldc leading dimension of C.
 **/
void matmul_16_6_1(float const *a,
                   float const *b,
                   float *c,
                   int64_t lda,
                   int64_t ldb,
                   int64_t ldc);

/**
 * @param a pointer to column-major matrix A.
 * @param b pointer to column-major matrix B.
 * @param c pointer to column-major matrix C.
 * @param lda leading dimension of A.
 * @param ldb leading dimension of B.
 * @param ldc leading dimension of C.
 **/
void matmul_16_6_64(float const *a,
                    float const *b,
                    float *c,
                    int64_t lda,
                    int64_t ldb,
                    int64_t ldc);

/**
 * @param a pointer to column-major matrix A.
 * @param b pointer to column-major matrix B.
 * @param c pointer to column-major matrix C.
 * @param lda leading dimension of A.
 * @param ldb leading dimension of B.
 * @param ldc leading dimension of C.
 **/
void matmul_64_6_64(float const *a,
                    float const *b,
                    float *c,
                    int64_t lda,
                    int64_t ldb,
                    int64_t ldc);
}

void gemm_ref(float const *i_a,
              float const *i_b,
              float *io_c,
              int64_t i_m,
              int64_t i_n,
              int64_t i_k,
              int64_t i_lda,
              int64_t i_ldb,
              int64_t i_ldc) {
    for (int l_m = 0; l_m < i_m; l_m++) {
        for (int l_n = 0; l_n < i_n; l_n++) {
            for (int l_k = 0; l_k < i_k; l_k++) {
                io_c[(l_n * i_ldc) + l_m] += i_a[(l_k * i_lda) + l_m] * i_b[(l_n * i_ldb) + l_k];
            }
        }
    }
}

float checkDif(float *arr_1,
               float *arr_2,
               int length) {
    float result = 0.0;

    for (int i = 0; i < length; i++) {
        if (std::abs(arr_1[i] - arr_2[i]) > 0.0001) {
            std::cout << "ID " << i << ": " << arr_1[i] << " / " << arr_2[i] << std::endl;
            result = std::abs(arr_1[i] - arr_2[i]);
        }
    }
    if (result < 1.0E-5) {
        return 0.0;
    }

    return result;
}

void bench_mm(uint64_t i_reps,
              int i_m,
              int i_n,
              int i_k,
              int i_lda,
              int i_ldb,
              int i_ldc,
              void (*matmul_kernel)(float const *,
                                    float const *,
                                    float *,
                                    int64_t,
                                    int64_t,
                                    int64_t)) {
    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double> dur;
    uint64_t reps = i_reps;

    srand48(time(NULL));

    int l_m = i_m;
    int l_k = i_k;
    int l_n = i_n;
    int l_lda = i_lda;
    int l_ldb = i_ldb;
    int l_ldc = i_ldc;

    double l_g_flops = 2 * l_k * l_n * l_m;

    // initialize matrix
    float *l_a = (float *)malloc(l_k * l_lda * sizeof(float));
    float *l_b = (float *)malloc(l_ldb * l_n * sizeof(float));
    float *l_c_1 = (float *)malloc(l_ldc * l_n * sizeof(float));
    float *l_c_2 = (float *)malloc(l_ldc * l_n * sizeof(float));

    for (int i = 0; i < (l_k * l_lda); i++) {
        l_a[i] = (float)drand48();
    }

    for (int i = 0; i < (l_ldb * l_n); i++) {
        l_b[i] = (float)drand48();
    }

    for (int i = 0; i < (l_ldc * l_n); i++) {
        l_c_1[i] = (float)drand48();
    }

    for (int i = 0; i < (l_ldc * l_n); i++) {
        l_c_2[i] = l_c_1[i];
    }

    gemm_ref(l_a, l_b, l_c_1, l_m, l_n, l_k, l_lda, l_ldb, l_ldc);
    matmul_kernel(l_a, l_b, l_c_2, l_lda, l_ldb, l_ldc);

    std::cout << "Error:  " << checkDif(l_c_1, l_c_2, l_m * l_n) << std::endl;

    start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < reps; i++) {
        matmul_kernel(l_a, l_b, l_c_2, l_lda, l_ldb, l_ldc);
    }
    end = std::chrono::steady_clock::now();

    dur = std::chrono::duration_cast<std::chrono::duration<double> >(end - start);

    std::cout << "M = " << l_m << " , K = " << l_k << " , N = " << l_n << std::endl;
    std::cout << "executions: " << reps << std::endl;
    std::cout << "duration: " << dur.count() << " seconds" << std::endl;

    l_g_flops *= reps;
    l_g_flops *= 1.0E-9;
    l_g_flops /= dur.count();

    std::cout << "GFLOPS: " << l_g_flops << std::endl;
    std::cout << "***********************" << std::endl;

    free(l_a);
    free(l_b);
    free(l_c_1);
    free(l_c_2);
}

int main() {
    bench_mm(15000000,
             16,
             6,
             1,
             16,
             1,
             16,
             matmul_16_6_1);

    bench_mm(1500000,
             16,
             6,
             64,
             16,
             64,
             16,
             matmul_16_6_64);

    bench_mm(700000,
             64,
             6,
             64,
             64,
             64,
             64,
             matmul_64_6_64);

    return EXIT_SUCCESS;
}
