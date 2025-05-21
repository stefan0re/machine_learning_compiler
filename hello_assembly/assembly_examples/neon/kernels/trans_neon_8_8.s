
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

    // Load 8 rows, 2 x q-registers per row (left and right halves)
    ld1 {v0.4s}, [x0], #16
    ld1 {v1.4s}, [x0], #16
    ld1 {v2.4s}, [x0], #16
    ld1 {v3.4s}, [x0], #16
    ld1 {v4.4s}, [x0], #16
    ld1 {v5.4s}, [x0], #16
    ld1 {v6.4s}, [x0], #16
    ld1 {v7.4s}, [x0], #16

    ld1 {v8.4s},  [x0], #16
    ld1 {v9.4s},  [x0], #16
    ld1 {v10.4s}, [x0], #16
    ld1 {v11.4s}, [x0], #16
    ld1 {v12.4s}, [x0], #16
    ld1 {v13.4s}, [x0], #16
    ld1 {v14.4s}, [x0], #16
    ld1 {v15.4s}, [x0], #16

    // v0–v7  = rows 0–3, left halves
    // v8–v15 = rows 4–7, left halves

    // Transpose pairs: 32-bit interleave (TRN1/TRN2)
    trn1 v16.4s, v0.4s, v1.4s
    trn2 v17.4s, v0.4s, v1.4s
    trn1 v18.4s, v2.4s, v3.4s
    trn2 v19.4s, v2.4s, v3.4s

    trn1 v20.4s, v4.4s, v5.4s
    trn2 v21.4s, v4.4s, v5.4s
    trn1 v22.4s, v6.4s, v7.4s
    trn2 v23.4s, v6.4s, v7.4s

    // Now 64-bit interleave (TRN1/TRN2 at 64-bit)
    trn1 v24.2d, v16.2d, v18.2d
    trn2 v25.2d, v16.2d, v18.2d
    trn1 v26.2d, v17.2d, v19.2d
    trn2 v27.2d, v17.2d, v19.2d

    trn1 v28.2d, v20.2d, v22.2d
    trn2 v29.2d, v20.2d, v22.2d
    trn1 v30.2d, v21.2d, v23.2d
    trn2 v31.2d, v21.2d, v23.2d

    // v24–v31 now contain the 8 columns of the transposed matrix

    // Store transposed columns
    st1 {v24.4s}, [x1], #16
    st1 {v25.4s}, [x1], #16
    st1 {v26.4s}, [x1], #16
    st1 {v27.4s}, [x1], #16
    st1 {v28.4s}, [x1], #16
    st1 {v29.4s}, [x1], #16
    st1 {v30.4s}, [x1], #16
    st1 {v31.4s}, [x1], #16

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

ret




