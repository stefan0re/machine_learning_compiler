#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>

extern "C" {
/**
 * @param a pointer to column-major matrix A.
 * @param b pointer to column-major matrix B.
 * @param c pointer to column-major matrix C.
 * @param lda leading dimension of A.
 * @param ldb leading dimension of B.
 * @param ldc leading dimension of C.
 **/
void matmul_16_6_1(float const* a,
                   float const* b,
                   float* c,
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
void matmul_16_6_64(float const* a,
                    float const* b,
                    float* c,
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
void matmul_64_6_64(float const* a,
                    float const* b,
                    float* c,
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
void matmul_64_48_64(float const* a,
                     float const* b,
                     float* c,
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
void matmul_14_6_64(float const* a,
                    float const* b,
                    float* c,
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
void matmul_15_6_64(float const* a,
                    float const* b,
                    float* c,
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
void matmul_64_64_64(float const* a,
                     float const* b,
                     float* c,
                     int64_t lda,
                     int64_t ldb,
                     int64_t ldc);

/**
 * @brief Identity primitive that transposes an 8x8 matrix.
 * @param a    Pointer to column-major matrix A.
 * @param b    Pointer to row-major matrix B.
 * @param ld_a Leading dimension of A.
 * @param ld_b Leading dimension of B.
 **/
void trans_neon_8_8(float const* a,
                    float* b,
                    int64_t ld_a,
                    int64_t ld_b);
}

void reference_mat_mul(float const* a,
                       float const* b,
                       float* c,
                       int64_t n,
                       int64_t m,
                       int64_t k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                c[(j * m) + i] += a[(l * m) + i] * b[(j * k) + l];
            }
        }
    }
}

void reference_transpose(float const* a, float* b, int64_t lda, int64_t ldb) {
    for (int64_t i = 0; i < lda; ++i) {
        for (int64_t j = 0; j < lda; ++j) {
            b[j * ldb + i] = a[i * lda + j];
        }
    }
}

void visualize_matix(float const* c,
                     int64_t height,
                     int64_t width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = j * height + i;
            std::cout << c[index] << " ";
        }
        std::cout << std::endl;
    }
}

void get_matrices(float* a,
                  float* b,
                  float* c,
                  float* c_ref,
                  int64_t n,
                  int64_t m,
                  int64_t k,
                  bool visualization = false) {
    float MAX = 100.f;
    // fill a
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            int a_index = j * m + i;
            a[a_index] = (1 - (double)visualization) * static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / MAX)) + (double)visualization * i;
        }
    }

    // fill b
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            int b_index = j * k + i;
            b[b_index] = (1 - (double)visualization) * static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / MAX)) + (double)visualization * j;
        }
    }

    // fill c
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int c_index = j * m + i;
            float element = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / MAX));
            c[c_index] = (1 - (double)visualization) * element;
            c_ref[c_index] = (1 - (double)visualization) * element;
        }
    }
}

int test_matmul(int64_t n,
                int64_t m,
                int64_t k,
                int64_t ops_per_call,
                int64_t iterations,
                void (*matmul_func)(float const*, float const*, float*, int64_t, int64_t, int64_t)) {
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Testing matmul_" << m << "_" << n << "_" << k << " ..." << std::endl;
    alignas(16) float a[k * m];
    alignas(16) float b[n * k];
    alignas(16) float c[n * m];
    alignas(16) float c_ref[n * m];
    std::chrono::_V2::system_clock::time_point start, end;
    bool is_correct = true;

    get_matrices(a, b, c, c_ref, n, m, k);

    reference_mat_mul(a, b, c_ref, n, m, k);
    matmul_func(a, b, c, m, k, m);

    double epsilon = 1e-3;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int c_index = j * m + i;
            if (std::fabs(c[c_index] - c_ref[c_index]) > (epsilon * std::fabs(c_ref[c_index]))) {
                std::cout << "Failed in: i=" << i << ", j=" << j << std::endl;
                std::cout << c[c_index] << " != " << c_ref[c_index] << ", Diff=" << (std::fabs(c[c_index] - c_ref[c_index])) << std::endl;
                return 0;
            }
        }
    }

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        matmul_func(a, b, c, m, k, m);
    }
    end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();
    double throughput = ((double)iterations * (double)ops_per_call) / duration;  // 192 flops in one iter

    std::cout << "\nIterations:\t" << iterations << " times" << std::endl;
    std::cout << "Duration:\t" << duration << " sec" << std::endl;
    std::cout << "Throughput:\t" << throughput / 1e9 << " GFLOPS\n"
              << std::endl;

    return 1;
}

int test_transpose_8_8(int64_t ops_per_call, int64_t iterations) {
    const int size = 8;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Testing transpose_8_8" << std::endl;
    alignas(16) float a[size * size];
    alignas(16) float b[size * size];
    alignas(16) float b_ref[size * size];
    alignas(16) float _temp[size * size];

    std::chrono::_V2::system_clock::time_point start, end;
    bool is_correct = true;

    get_matrices(a, b, b_ref, _temp, size, size, size);

    reference_transpose(a, b_ref, size, size);
    trans_neon_8_8(a, b, size, size);

    double epsilon = 1e-3;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int b_index = j * size + i;
            if (std::fabs(b[b_index] - b_ref[b_index]) > (epsilon * std::fabs(b_ref[b_index]))) {
                std::cout << "Failed in: i=" << i << ", j=" << j << std::endl;
                std::cout << b[b_index] << " != " << b_ref[b_index] << ", Diff=" << (std::fabs(b[b_index] - b_ref[b_index])) << std::endl;
                // DEBUG
                // return 0;
            }
        }
    }

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        trans_neon_8_8(a, b, size, size);
    }
    end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();
    double throughput = ((double)iterations * (double)ops_per_call) / duration;

    std::cout << "\nIterations:\t" << iterations << " times" << std::endl;
    std::cout << "Duration:\t" << duration << " sec" << std::endl;
    std::cout << "Throughput:\t" << throughput / 1e9 << " GFLOPS\n"
              << std::endl;

    return 1;
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    if (!test_matmul(6, 16, 1, 192, 150000000, matmul_16_6_1)) {
        return 1;
    }
    if (!test_matmul(6, 16, 64, 192 * 64, 10000000, matmul_16_6_64)) {
        return 1;
    }
    if (!test_matmul(6, 64, 64, 192 * 64 * 4, 2000000, matmul_64_6_64)) {
        return 1;
    }
    if (!test_matmul(48, 64, 64, 192 * 64 * 4 * 8, 250000, matmul_64_48_64)) {
        return 1;
    }
    if (!test_matmul(6, 14, 64, 192 * 64, 10000000, matmul_14_6_64)) {
        return 1;
    }
    if (!test_matmul(6, 15, 64, 192 * 64, 10000000, matmul_15_6_64)) {
        return 1;
    }
    if (!test_matmul(64, 64, 64, 128 * 64 * 8 * 8, 150000, matmul_64_64_64)) {
        return 1;
    }

    // TEST TRANSPOSE
    // -------------------------------------------------
    if (!test_transpose_8_8(1, 1)) {
        return 1;
    }

    return 0;
}