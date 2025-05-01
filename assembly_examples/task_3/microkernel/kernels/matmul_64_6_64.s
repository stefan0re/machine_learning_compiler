    .text
    .type matmul_64_6_64, %function
    .global matmul_64_6_64

matmul_64_6_64:
    // Parameter/Result Registers:
    // x0 - a pointer to column-major matrix A.
    // x1 - b pointer to column-major matrix B.
    // x2 - c pointer to column-major matrix C.
    // x3 - lda leading dimension of A.
    // x4 - ldb leading dimension of B.
    // x5 - ldc leading dimension of C.

    // save frame pointer and link register
    stp fp, lr, [sp, #-16]!

    // update frame pointer to current stack pointer
    mov fp, sp

    // save callee-saved registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // working pointer to C
    mov x8, x2
    
    add x9, xzr, x3, lsl #2 // byte offset for lda
    add x10, xzr, x4, lsl #2 // byte offset for ldb
    add x11, xzr, x5, lsl #2 // byte offset for ldc
    

    mov x16, x3 // loop counter to m
    mov x6, x0 // working pointer to A
    mov x12, xzr // a tile counter

m_loop:
    sub x16, x16, #16

    // load c
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x8], x11
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x8], x11
    ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x8], x11
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x8], x11
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x8], x11
    ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x8]

    mov x17, x4 // loop counter to k
    mov x7, x1 // B row pointer 
    mov x13, #1 // B counter

k_loop:
    sub x17, x17, #1

    // load a 
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], x9

    // load b
    ld1 {v4.s}[0], [x7], x10
    ld1 {v4.s}[1], [x7], x10
    ld1 {v4.s}[2], [x7], x10
    ld1 {v4.s}[3], [x7], x10
    ld1 {v5.s}[0], [x7], x10
    ld1 {v5.s}[1], [x7]

    // matrix multiplication

    // column 0
    fmla v8.4s, v0.4s, v4.S[0]
    fmla v9.4s, v1.4s, v4.S[0]
    fmla v10.4s, v2.4s, v4.S[0]
    fmla v11.4s, v3.4s, v4.S[0]

    // column 1
    fmla v12.4s, v0.4s, v4.S[1]
    fmla v13.4s, v1.4s, v4.S[1]
    fmla v14.4s, v2.4s, v4.S[1]
    fmla v15.4s, v3.4s, v4.S[1]

    // column 2
    fmla v16.4s, v0.4s, v4.S[2]
    fmla v17.4s, v1.4s, v4.S[2]
    fmla v18.4s, v2.4s, v4.S[2]
    fmla v19.4s, v3.4s, v4.S[2]

    // column 3
    fmla v20.4s, v0.4s, v4.S[3]
    fmla v21.4s, v1.4s, v4.S[3]
    fmla v22.4s, v2.4s, v4.S[3]
    fmla v23.4s, v3.4s, v4.S[3]

    // column 4
    fmla v24.4s, v0.4s, v5.S[0]
    fmla v25.4s, v1.4s, v5.S[0]
    fmla v26.4s, v2.4s, v5.S[0]
    fmla v27.4s, v3.4s, v5.S[0]

    // column 5
    fmla v28.4s, v0.4s, v5.S[1]
    fmla v29.4s, v1.4s, v5.S[1]
    fmla v30.4s, v2.4s, v5.S[1]
    fmla v31.4s, v3.4s, v5.S[1]

    // move one row down in B
    mov x7, x1
    add x7, x7, x13, lsl #2

    // increment B counter
    add x13, x13, #1

    cbnz x17, k_loop

    mov x8, x2
    add x8, x8, x12, lsl #2

    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x8], x9
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x8], x9
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x8], x9
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x8], x9
    st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x8], x9
    st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x8]

    mov x8, x2
    mov x6, x0
    // increment A counter
    add x12, x12, #16
    // move to new tile (move 16 rows down)
    add x8, x8, x12, lsl #2
    add x6, x6, x12, lsl #2

    cbnz x16, m_loop

    // restore callee-saved registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // restore frame pointer and link register
    ldp fp, lr, [sp], #16

    ret
