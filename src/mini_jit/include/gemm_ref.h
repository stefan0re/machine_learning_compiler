#ifndef INCLUDE_GEMM_REF
#define INCLUDE_GEMM_REF
#include <cstdint>
#include <cmath>


void gemm_ref( float        const * i_a,
               float        const * i_b,
               float              * io_c,
               int64_t              i_m,
               int64_t              i_n,
               int64_t              i_k,
               int64_t              i_lda,
               int64_t              i_ldb,
               int64_t              i_ldc );

#endif