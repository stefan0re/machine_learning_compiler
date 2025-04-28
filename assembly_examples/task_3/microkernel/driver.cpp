#include <chrono>
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

void visualize_matix(float* c,
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

int test_matmul_16_6_64() {
    alignas(16) float a[16 * 64];
    alignas(16) float b[64 * 6];
    alignas(16) float c[16 * 6];
    alignas(16) float c_ref[16 * 6];
    std::chrono::_V2::system_clock::time_point start, end;
    bool is_correct = true;
    int iterations = 100000000;

    // fill a
    for (int j = 0; j < 64; j++) {
        for (int i = 0; i < 16; i++) {
            int a_index = j * 16 + i;
            a[a_index] = (float)i;
        }
    }

    // fill b
    for (int j = 0; j < 6; j++) {
        for (int i = 0; i < 64; i++) {
            int b_index = j * 64 + i;
            b[b_index] = (float)j;
        }
    }

    // fill c
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 6; j++) {
            int c_index = j * 16 + i;
            c[c_index] = 0;
            c_ref[c_index] = 0;
        }
    }

    matmul_16_6_64(a, b, c, 16, 6, 16);
    visualize_matix(c, 16, 6);
    reference_mat_mul(a, b, c_ref, 6, 16, 64);
    std::cout << std::endl;
    visualize_matix(c_ref, 16, 6);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 6; j++) {
            int c_index = j * 16 + i;
            if (c[c_index] != c_ref[c_index]) {
                std::cout << "Failed in matmul_16_6_1: i=" << i << ", j=" << j << std::endl;

                std::cout << c[c_index] << " != " << c_ref[c_index] << std::endl;
                is_correct = false;
            }
        }
    }

    /*start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        matmul_16_6_1(a, b, c, 16, 6, 16);
    }
    end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();
    double throughput = (iterations / duration) * 192;  // 192 flops in one iter

    std::cout << "---------------------------------" << std::endl;
    std::cout << "matmul_16_6_1" << std::endl;
    std::cout << "Duration:\t" << duration << " sec" << std::endl;
    std::cout << "Throughput:\t" << throughput / 1e9 << " GFLOPS\n"
              << std::endl;*/

    return is_correct;
}

int test_matmul_16_6_1() {
    alignas(16) float a[16];
    alignas(16) float b[6];
    alignas(16) float c[16 * 6];
    alignas(16) float c_ref[16 * 6];
    std::chrono::_V2::system_clock::time_point start, end;
    bool is_correct = true;
    int iterations = 100000000;

    for (int i = 0; i < 16; i++) {
        a[i] = (float)i;
        for (int j = 0; j < 6; j++) {
            b[j] = (float)j;
            int c_index = j * 16 + i;
            c[c_index] = 0;
            c_ref[c_index] = 0;
        }
    }

    matmul_16_6_1(a, b, c, 16, 6, 16);
    reference_mat_mul(a, b, c_ref, 6, 16, 1);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 6; j++) {
            int c_index = j * 16 + i;
            if (c[c_index] != c_ref[c_index]) {
                std::cout << "Failed in matmul_16_6_1: i=" << i << ", j=" << j << std::endl;

                std::cout << c[c_index] << " != " << c_ref[c_index] << std::endl;
                is_correct = false;
            }
        }
    }

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        matmul_16_6_1(a, b, c, 16, 6, 16);
    }
    end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();
    double throughput = (iterations / duration) * 192;  // 192 flops in one iter

    std::cout << "---------------------------------" << std::endl;
    std::cout << "matmul_16_6_1" << std::endl;
    std::cout << "Duration:\t" << duration << " sec" << std::endl;
    std::cout << "Throughput:\t" << throughput / 1e9 << " GFLOPS\n"
              << std::endl;

    return is_correct;
}

int main() {
    if (!test_matmul_16_6_1()) {
        return 1;
    }
    if (!test_matmul_16_6_64()) {
        return 1;
    }

    return 0;
}
