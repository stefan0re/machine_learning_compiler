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

    // save stack
    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // load working register
    // A
    mov x7, x0
    // B
    mov x8, x1

    // load A
    ld1     {v0.4s-v3.4s},   [x7], #64
    ld1     {v4.4s-v7.4s},   [x7], #64
    ld1     {v16.4s-v19.4s}, [x7], #64
    ld1     {v20.4s-v23.4s}, [x7], #64

    // top half (shift 32-bit values)
    trn1    v24.4s, v0.4s, v2.4s    // row0
    trn1    v25.4s, v1.4s, v3.4s
    trn2    v26.4s, v0.4s, v2.4s    // row1
    trn2    v27.4s, v1.4s, v3.4s
    trn1    v28.4s, v4.4s, v6.4s    // row2
    trn1    v29.4s, v5.4s, v7.4s
    trn2    v30.4s, v4.4s, v6.4s    // row3
    trn2    v31.4s, v5.4s, v7.4s
    
    // bottom half (shift 32-bit values)
    trn1    v0.4s, v16.4s, v18.4s   // row4
    trn1    v1.4s, v17.4s, v19.4s
    trn2    v2.4s, v16.4s, v18.4s   // row5
    trn2    v3.4s, v17.4s, v19.4s
    trn1    v4.4s, v20.4s, v22.4s   // row6
    trn1    v5.4s, v21.4s, v23.4s
    trn2    v6.4s, v20.4s, v22.4s   // row7
    trn2    v7.4s, v21.4s, v23.4s

    // top half (shift 64-bit values) 
    trn1    v16.2d, v24.2d, v28.2d  // row0a
    trn1    v17.2d, v0.2d, v4.2d    // row0b
    trn1    v18.2d, v26.2d, v30.2d  // row1a
    trn1    v19.2d, v2.2d, v6.2d    // row1b
    trn2    v20.2d, v24.2d, v28.2d  // row2a
    trn2    v21.2d, v0.2d, v4.2d    // row2b
    trn2    v22.2d, v26.2d, v30.2d  // row3a
    trn2    v23.2d, v2.2d, v6.2d    // row3b

    // save to reuse registers 
    st1     {v16.4s-v19.4s}, [x8], #64
    st1     {v20.4s-v23.4s}, [x8], #64

    // bottom half (shift 64-bit values)
    trn1    v16.2d, v25.2d, v29.2d  // row4a
    trn1    v17.2d, v1.2d, v5.2d    // row4b
    trn1    v18.2d, v27.2d, v31.2d  // row5a
    trn1    v19.2d, v3.2d, v7.2d    // row5b
    trn2    v20.2d, v25.2d, v29.2d  // row4a
    trn2    v21.2d, v1.2d, v5.2d    // row4b
    trn2    v22.2d, v27.2d, v31.2d  // row5a
    trn2    v23.2d, v3.2d, v7.2d    // row5b

    // store B
    st1     {v16.4s-v19.4s}, [x8], #64
    st1     {v20.4s-v23.4s}, [x8]

    // restore stack
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

ret
