
/**
* @brief Identity primitive that transposes an 8x8 matrix A and return it in B.
* @param a    Pointer to column-major matrix A.
* @param b    Pointer to row-major matrix B.
* @param ld_a Leading dimension of A.
* @param ld_b Leading dimension of B.
* static void trans_neon_8_8(float const* a, float* b, int64_t ld_a, int64_t ld_b);
**/


    .text
    .type trans_neon_8_8, %function
    .global trans_neon_8_8
trans_neon_8_8:

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // Load A
    mov x7, x0             // x0 = input matrix pointer
    ld1 {v0.4s}, [x7], #16
    ld1 {v1.4s}, [x7], #16
    ld1 {v2.4s}, [x7], #16
    ld1 {v3.4s}, [x7], #16
    ld1 {v4.4s}, [x7], #16
    ld1 {v5.4s}, [x7], #16
    ld1 {v6.4s}, [x7], #16
    ld1 {v7.4s}, [x7], #16

    // Transpose: 8x8 of 32-bit values
    trn1 v8.4s, v0.4s, v1.4s
    trn2 v9.4s, v0.4s, v1.4s
    trn1 v10.4s, v2.4s, v3.4s
    trn2 v11.4s, v2.4s, v3.4s
    trn1 v12.4s, v4.4s, v5.4s
    trn2 v13.4s, v4.4s, v5.4s
    trn1 v14.4s, v6.4s, v7.4s
    trn2 v15.4s, v6.4s, v7.4s

    zip1 v0.2d, v8.2d, v10.2d
    zip2 v1.2d, v8.2d, v10.2d
    zip1 v2.2d, v9.2d, v11.2d
    zip2 v3.2d, v9.2d, v11.2d
    zip1 v4.2d, v12.2d, v14.2d
    zip2 v5.2d, v12.2d, v14.2d
    zip1 v6.2d, v13.2d, v15.2d
    zip2 v7.2d, v13.2d, v15.2d

    // Store A
    mov x8, x1             // x1 = output matrix pointer
    st1 {v0.4s}, [x8], #16
    st1 {v1.4s}, [x8], #16
    st1 {v2.4s}, [x8], #16
    st1 {v3.4s}, [x8], #16
    st1 {v4.4s}, [x8], #16
    st1 {v5.4s}, [x8], #16
    st1 {v6.4s}, [x8], #16
    st1 {v7.4s}, [x8], #16


    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

ret
