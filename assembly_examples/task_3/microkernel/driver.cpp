#include <iostream>
#include <chrono>

extern "C"{
            /**
            * @param a pointer to column-major matrix A.
            * @param b pointer to column-major matrix B.
            * @param c pointer to column-major matrix C.
            * @param lda leading dimension of A.
            * @param ldb leading dimension of B.
            * @param ldc leading dimension of C.
            **/
            void matmul_16_6_1( float   const * a,
                                float   const * b,
                                float         * c,
                                int64_t         lda,
                                int64_t         ldb,
                                int64_t         ldc );
}

void reference_mat_mul( float   const * a,
                        float   const * b,
                        float         * c,
                        int64_t         lda,
                        int64_t         ldb,
                        int64_t         ldc ) {
    
    for (int j = 0; j < 6; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < 16; ++i) {
            int c_index = j * ldb + i;
            c[c_index] += a[i] * b[j];
        }
    }
}

int test_matmul_16_6_1() {
    alignas(16) float a[16];
    alignas(16) float b[6];
    alignas(16) float c[16 * 6];
    alignas(16) float c_ref[16 * 6];
    std::chrono::_V2::system_clock::time_point start, end;

    for (int i = 0; i < 16; i++) {
        a[i] = (float) i;
        for (int j = 0; j < 6; j++) {
            b[j] = (float) j;
            int c_index = j * 16 + i;
            c[c_index] = 0;
            c_ref[c_index] = 0;
        }
    }

    start = std::chrono::high_resolution_clock::now();
    matmul_16_6_1(a, b, c, 16, 6, 16);
    end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();
    double throughput = (24 / duration) * 8;  // 8 ops in one iter

    std::cout << "---------------------------------" << std::endl;
    std::cout << "matmul_16_6_1" << std::endl;
    std::cout << "Duration:\t" << duration << " sec" << std::endl;
    std::cout << "Throughput:\t" << throughput / 1e9 << " GOPS\n"
              << std::endl;

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 6; j++) {
            int c_index = j * 16 + i;
            if (c[c_index] != c_ref[c_index]) {
                std::cout << "Failed in matmul_16_6_1: i=" << i << ", j=" << j << std::endl;
                return 0;
            }
        }
    }
    return 1;
}

int main() {
    if (test_matmul_16_6_1()) {
        std::cout << "matmul_16_6_1 was executed correctly" << std::endl;
    } else {
        return 0;
    }

    return 1;
}
