#include <cstdint>
#include <cmath>

#include "gemm_ref.h"


void gemm_ref( float        const * i_a,
               float        const * i_b,
               float              * io_c,
               int64_t              i_m,
               int64_t              i_n,
               int64_t              i_k,
               int64_t              i_lda,
               int64_t              i_ldb,
               int64_t              i_ldc ){
    for( int l_m = 0; l_m < i_m; l_m++ ) {
        for( int l_n = 0; l_n < i_n; l_n++ ) {
            for( int l_k = 0; l_k < i_k; l_k++ ) {
                io_c[ (l_n*i_ldc) + l_m ] += i_a[ (l_k*i_lda) + l_m ] * i_b[ (l_n*i_ldb) + l_k ];
            }
        }
    }
}