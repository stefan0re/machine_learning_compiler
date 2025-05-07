    .text
    .type matmul_64_64_64, %function
    .global matmul_64_64_64
matmul_64_64_64:

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // working pointers
    mov x7, x0 // A
    mov x8, x1 // B
    mov x9, x2 // C

    // adjust leading dimension
    lsl x3, x3, #2 // lda
    lsl x4, x4, #2 // ldb
    lsl x5, x5, #2 // ldc

    // B block offset
    mov x13, #8
    mul x13, x13, x4

    // C block offset
    mov x14, #8
    mul x14, x14, x5

    // A reset point
    mov x15, x0

    // N loop counter
    mov x12, #8

n_loop:
    sub x12, x12, #1

    // M loop counter
    mov x11, #8

m_loop:
    sub x11, x11, #1

    // load C 
    ld1 {v0.4s, v1.4s}, [x9]
    add x9, x9, x5
    ld1 {v2.4s, v3.4s}, [x9]
    add x9, x9, x5
    ld1 {v4.4s, v5.4s}, [x9]
    add x9, x9, x5
    ld1 {v6.4s, v7.4s}, [x9]
    add x9, x9, x5
    ld1 {v8.4s, v9.4s}, [x9]
    add x9, x9, x5
    ld1 {v10.4s, v11.4s}, [x9]
    add x9, x9, x5
    ld1 {v12.4s, v13.4s}, [x9]
    add x9, x9, x5
    ld1 {v14.4s, v15.4s}, [x9]

    // K loop counter
    mov x10, #64

k_loop:
    sub x10, x10, #1

    // load A
    ld1 {v16.4s, v17.4s}, [x7]
    add x7, x7, x3

    // load B
    ldr s18, [x8]
    add x8, x8, x4
    ldr s19, [x8]
    add x8, x8, x4
    ldr s20, [x8]
    add x8, x8, x4
    ldr s21, [x8]
    add x8, x8, x4

    ldr s22, [x8]    
    add x8, x8, x4
    ldr s23, [x8]   
    add x8, x8, x4
    ldr s24, [x8]    
    add x8, x8, x4
    ldr s25, [x8]    
    add x8, x8, x4

    fmla v0.4s, v16.4s, v18.s[0]
    fmla v1.4s, v17.4s, v18.s[0]
    fmla v2.4s, v16.4s, v19.s[0]
    fmla v3.4s, v17.4s, v19.s[0]
    fmla v4.4s, v16.4s, v20.s[0]
    fmla v5.4s, v17.4s, v20.s[0]
    fmla v6.4s, v16.4s, v21.s[0]
    fmla v7.4s, v17.4s, v21.s[0]

    fmla v8.4s, v16.4s, v22.s[0]
    fmla v9.4s, v17.4s, v22.s[0]
    fmla v10.4s, v16.4s, v23.s[0]
    fmla v11.4s, v17.4s, v23.s[0]
    fmla v12.4s, v16.4s, v24.s[0]
    fmla v13.4s, v17.4s, v24.s[0]
    fmla v14.4s, v16.4s, v25.s[0]
    fmla v15.4s, v17.4s, v25.s[0]

    sub x8, x8, x13 // reset B-Block
    add x8, x8, #4 // advance to next B-row

    cbnz x10, k_loop
 
    mov x9, x2 // reset C-Pointer

    st1 {v0.4s, v1.4s}, [x9]
    add x9, x9, x5
    st1 {v2.4s, v3.4s}, [x9]
    add x9, x9, x5
    st1 {v4.4s, v5.4s}, [x9]
    add x9, x9, x5
    st1 {v6.4s, v7.4s}, [x9]
    add x9, x9, x5
    st1 {v8.4s, v9.4s}, [x9]
    add x9, x9, x5
    st1 {v10.4s, v11.4s}, [x9]
    add x9, x9, x5
    st1 {v12.4s, v13.4s}, [x9]
    add x9, x9, x5
    st1 {v14.4s, v15.4s}, [x9]

    // adjust origin A and C pointer to next column
    add x0, x0, #32
    add x2, x2, #32

    mov x7, x0
    mov x8, x1
    mov x9, x2

    cbnz x11, m_loop

    sub x2, x2, #32*8

    // set A
    mov x0, x15
    mov x7, x15

    // set B
    add x1, x1, x13
    mov x8, x1

    // set C
    add x2, x2, x14
    mov x9, x2

    cbnz x12, n_loop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ret