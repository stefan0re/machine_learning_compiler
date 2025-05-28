#ifndef INCLUDE_GEMM_REF
#define INCLUDE_GEMM_REF
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
              int64_t i_ldc);
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
                int64_t i_br_stride_b);
#endif