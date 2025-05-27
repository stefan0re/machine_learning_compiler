#include "gemm_ref.h"

#include <cmath>
#include <cstdint>

void gemm_ref(float const* i_a,
              float const* i_b,
              float* io_c,
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

void brgemm_ref(float const* i_a,
                float const* i_b,
                float* io_c,
                int64_t i_m,
                int64_t i_n,
                int64_t i_k_gemm,
                int64_t i_k_br,
                int64_t i_lda,
                int64_t i_ldb,
                int64_t i_ldc,
                int64_t i_br_stride_a,
                int64_t i_br_stride_b) {
    for (int64_t l_k_br = 0; l_k_br < i_k_br; l_k_br++) {
        for (int64_t l_m = 0; l_m < i_m; l_m++) {
            for (int64_t l_n = 0; l_n < i_n; l_n++) {
                for (int64_t l_k_gemm = 0; l_k_gemm < i_k_gemm; l_k_gemm++) {
                    io_c[(l_n * i_ldc) + l_m] += i_a[(l_k_br * i_br_stride_a) + (l_k_gemm * i_lda) + l_m] * i_b[(l_k_br * i_br_stride_b) + (l_n * i_k_gemm) + l_k_gemm];
                }
            }
        }
    }
}