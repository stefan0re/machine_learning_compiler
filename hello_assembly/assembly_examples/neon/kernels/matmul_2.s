    .type matmul_14_6_64, %function
    .global matmul_14_6_64
matmul_14_6_64:

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // store pointers
    mov x7, x0 // A
    mov x8, x1 // B
    mov x9, x2 // C

    // adjust leading dimension
    lsl x3, x3, #2 // A
    lsl x4, x4, #2 // B
    lsl x5, x5, #2 // C

    // set to size of one B block 64*6
    mov x15, #5
    mul x15, x15, x4

    // K loop counter
    mov x10, #64

    // load C 
    // load column 1
    ld1 {v0.4s}, [x9]
    add x9, x9, #16
    ld1 {v1.4s}, [x9]
    add x9, x9, #16
    ld1 {v2.4s}, [x9]
    add x9, x9, #16
    ld1 {v3.2s}, [x9]
    add x9, x9, #8

    // load column 2
    ld1 {v4.4s}, [x9]
    add x9, x9, #16
    ld1 {v5.4s}, [x9]
    add x9, x9, #16
    ld1 {v6.4s}, [x9]
    add x9, x9, #16
    ld1 {v7.2s}, [x9]
    add x9, x9, #8
    
    // load column 3
    ld1 {v8.4s}, [x9]
    add x9, x9, #16
    ld1 {v9.4s}, [x9]
    add x9, x9, #16
    ld1 {v10.4s}, [x9]
    add x9, x9, #16
    ld1 {v11.2s}, [x9]
    add x9, x9, #8
    
    // load column 4
    ld1 {v12.4s}, [x9]
    add x9, x9, #16
    ld1 {v13.4s}, [x9]
    add x9, x9, #16
    ld1 {v14.4s}, [x9]
    add x9, x9, #16
    ld1 {v15.2s}, [x9]
    add x9, x9, #8

    // load column 5
    ld1 {v16.4s}, [x9]
    add x9, x9, #16
    ld1 {v17.4s}, [x9]
    add x9, x9, #16
    ld1 {v18.4s}, [x9]
    add x9, x9, #16
    ld1 {v19.2s}, [x9]
    add x9, x9, #8
    
    // load column 6
    ld1 {v20.4s}, [x9]
    add x9, x9, #16
    ld1 {v21.4s}, [x9]
    add x9, x9, #16
    ld1 {v22.4s}, [x9]
    add x9, x9, #16
    ld1 {v23.2s}, [x9]
    add x9, x9, #8


k_loop:
    sub x10, x10, #1

    // load A
    ld1 {v24.4s}, [x0] 
    add x0, x0, #16
    ld1 {v25.4s}, [x0]
    add x0, x0, #16
    ld1 {v26.4s}, [x0]
    add x0, x0, #16
    ld1 {v27.2s}, [x0]
    add x0, x0, #8

    // load first 4 numbers of B
    ldr s28, [x1]
    add x1, x1, x4
    ldr s29, [x1]
    add x1, x1, x4

    ldr s30, [x1]
    add x1, x1, x4
    ldr s31, [x1]
    add x1, x1, x4

    // calculations on first 2 numbers of B
    fmla v0.4s, v24.4s, v28.s[0]
    fmla v1.4s, v25.4s, v28.s[0]
    fmla v2.4s, v26.4s, v28.s[0]
    fmla v3.4s, v27.4s, v28.s[0]

    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.4s, v25.4s, v29.s[0]
    fmla v6.4s, v26.4s, v29.s[0]
    fmla v7.4s, v27.4s, v29.s[0]

    // load last 2 numbers of B
    ldr s28, [x1]
    add x1, x1, x4
    ldr s29, [x1]

    // calculations on last 4 numbers of B
    fmla v8.4s, v24.4s, v30.s[0]
    fmla v9.4s, v25.4s, v30.s[0]
    fmla v10.4s, v26.4s, v30.s[0]
    fmla v11.4s, v27.4s, v30.s[0]

    fmla v12.4s, v24.4s, v31.s[0]
    fmla v13.4s, v25.4s, v31.s[0]
    fmla v14.4s, v26.4s, v31.s[0]
    fmla v15.4s, v27.4s, v31.s[0]

    fmla v16.4s, v24.4s, v28.s[0]
    fmla v17.4s, v25.4s, v28.s[0]
    fmla v18.4s, v26.4s, v28.s[0]
    fmla v19.4s, v27.4s, v28.s[0]

    fmla v20.4s, v24.4s, v29.s[0]
    fmla v21.4s, v25.4s, v29.s[0]
    fmla v22.4s, v26.4s, v29.s[0]
    fmla v23.4s, v27.4s, v29.s[0]

    // jump to next B row
    sub x1, x1, x15
    add x1, x1, #4

    cbnz x10, k_loop

    // store C
    mov x9, x2
 
    // store column 1
    st1 {v0.4s}, [x9]
    add x9, x9, #16
    st1 {v1.4s}, [x9]
    add x9, x9, #16
    st1 {v2.4s}, [x9]
    add x9, x9, #16
    st1 {v3.2s}, [x9]
    add x9, x9, #8

    // store column 2
    st1 {v4.4s}, [x9]
    add x9, x9, #16
    st1 {v5.4s}, [x9]
    add x9, x9, #16
    st1 {v6.4s}, [x9]
    add x9, x9, #16
    st1 {v7.2s}, [x9]
    add x9, x9, #8
    
    // store column 3
    st1 {v8.4s}, [x9]
    add x9, x9, #16
    st1 {v9.4s}, [x9]
    add x9, x9, #16
    st1 {v10.4s}, [x9]
    add x9, x9, #16
    st1 {v11.2s}, [x9]
    add x9, x9, #8
    
    // store column 4
    st1 {v12.4s}, [x9]
    add x9, x9, #16
    st1 {v13.4s}, [x9]
    add x9, x9, #16
    st1 {v14.4s}, [x9]
    add x9, x9, #16
    st1 {v15.2s}, [x9]
    add x9, x9, #8

    // store column 5
    st1 {v16.4s}, [x9]
    add x9, x9, #16
    st1 {v17.4s}, [x9]
    add x9, x9, #16
    st1 {v18.4s}, [x9]
    add x9, x9, #16
    st1 {v19.2s}, [x9]
    add x9, x9, #8
    
    // store column 6
    st1 {v20.4s}, [x9]
    add x9, x9, #16
    st1 {v21.4s}, [x9]
    add x9, x9, #16
    st1 {v22.4s}, [x9]
    add x9, x9, #16
    st1 {v23.2s}, [x9]
    add x9, x9, #8


    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ret

    .type matmul_15_6_64, %function
    .global matmul_15_6_64
matmul_15_6_64:

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // store pointers
    mov x7, x0 // A
    mov x8, x1 // B
    mov x9, x2 // C

    // adjust leading dimension
    lsl x3, x3, #2 // A
    lsl x4, x4, #2 // B
    lsl x5, x5, #2 // C

    // set to size of one B block 64*6
    mov x15, #5
    mul x15, x15, x4

    // K loop counter
    mov x10, #64

    // load C 
    // load column 1
    ld1 {v0.4s}, [x9]
    add x9, x9, #16
    ld1 {v1.4s}, [x9]
    add x9, x9, #16
    ld1 {v2.4s}, [x9]
    add x9, x9, #16
    ld1 {v3.2s}, [x9]
    add x9, x9, #8
    ld1 {v3.s}[2], [X9]
    add x9, x9, #4

    // load column 2
    ld1 {v4.4s}, [x9]
    add x9, x9, #16
    ld1 {v5.4s}, [x9]
    add x9, x9, #16
    ld1 {v6.4s}, [x9]
    add x9, x9, #16
    ld1 {v7.2s}, [x9]
    add x9, x9, #8
    ld1 {v7.s}[2], [X9]
    add x9, x9, #4
    
    // load column 3
    ld1 {v8.4s}, [x9]
    add x9, x9, #16
    ld1 {v9.4s}, [x9]
    add x9, x9, #16
    ld1 {v10.4s}, [x9]
    add x9, x9, #16
    ld1 {v11.2s}, [x9]
    add x9, x9, #8
    ld1 {v11.s}[2], [X9]
    add x9, x9, #4
    
    // load column 4
    ld1 {v12.4s}, [x9]
    add x9, x9, #16
    ld1 {v13.4s}, [x9]
    add x9, x9, #16
    ld1 {v14.4s}, [x9]
    add x9, x9, #16
    ld1 {v15.2s}, [x9]
    add x9, x9, #8
    ld1 {v15.s}[2], [X9]
    add x9, x9, #4

    // load column 5
    ld1 {v16.4s}, [x9]
    add x9, x9, #16
    ld1 {v17.4s}, [x9]
    add x9, x9, #16
    ld1 {v18.4s}, [x9]
    add x9, x9, #16
    ld1 {v19.2s}, [x9]
    add x9, x9, #8
    ld1 {v19.s}[2], [X9]
    add x9, x9, #4
    
    // load column 6
    ld1 {v20.4s}, [x9]
    add x9, x9, #16
    ld1 {v21.4s}, [x9]
    add x9, x9, #16
    ld1 {v22.4s}, [x9]
    add x9, x9, #16
    ld1 {v23.2s}, [x9]
    add x9, x9, #8
    ld1 {v23.s}[2], [X9]
    add x9, x9, #4


k1_loop:
    sub x10, x10, #1

    // load A
    ld1 {v24.4s}, [x0] 
    add x0, x0, #16
    ld1 {v25.4s}, [x0]
    add x0, x0, #16
    ld1 {v26.4s}, [x0]
    add x0, x0, #16
    ld1 {v27.2s}, [x0]
    add x0, x0, #8
    ld1 {v27.s}[2], [X0]
    add x0, x0, #4

    // load first 4 numbers of B
    ldr s28, [x1]
    add x1, x1, x4
    ldr s29, [x1]
    add x1, x1, x4

    ldr s30, [x1]
    add x1, x1, x4
    ldr s31, [x1]
    add x1, x1, x4

    // calculations on first 2 numbers of B
    fmla v0.4s, v24.4s, v28.s[0]
    fmla v1.4s, v25.4s, v28.s[0]
    fmla v2.4s, v26.4s, v28.s[0]
    fmla v3.4s, v27.4s, v28.s[0]

    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.4s, v25.4s, v29.s[0]
    fmla v6.4s, v26.4s, v29.s[0]
    fmla v7.4s, v27.4s, v29.s[0]

    // load last 2 numbers of B
    ldr s28, [x1]
    add x1, x1, x4
    ldr s29, [x1]

    // calculations on last 4 numbers of B
    fmla v8.4s, v24.4s, v30.s[0]
    fmla v9.4s, v25.4s, v30.s[0]
    fmla v10.4s, v26.4s, v30.s[0]
    fmla v11.4s, v27.4s, v30.s[0]

    fmla v12.4s, v24.4s, v31.s[0]
    fmla v13.4s, v25.4s, v31.s[0]
    fmla v14.4s, v26.4s, v31.s[0]
    fmla v15.4s, v27.4s, v31.s[0]

    fmla v16.4s, v24.4s, v28.s[0]
    fmla v17.4s, v25.4s, v28.s[0]
    fmla v18.4s, v26.4s, v28.s[0]
    fmla v19.4s, v27.4s, v28.s[0]

    fmla v20.4s, v24.4s, v29.s[0]
    fmla v21.4s, v25.4s, v29.s[0]
    fmla v22.4s, v26.4s, v29.s[0]
    fmla v23.4s, v27.4s, v29.s[0]

    // jump to next B row
    sub x1, x1, x15
    add x1, x1, #4

    cbnz x10, k1_loop

    // store C
    mov x9, x2
 
    // store column 1
    st1 {v0.4s}, [x9]
    add x9, x9, #16
    st1 {v1.4s}, [x9]
    add x9, x9, #16
    st1 {v2.4s}, [x9]
    add x9, x9, #16
    st1 {v3.2s}, [x9]
    add x9, x9, #8
    st1 {v3.s}[2], [X9]
    add x9, x9, #4

    // store column 2
    st1 {v4.4s}, [x9]
    add x9, x9, #16
    st1 {v5.4s}, [x9]
    add x9, x9, #16
    st1 {v6.4s}, [x9]
    add x9, x9, #16
    st1 {v7.2s}, [x9]
    add x9, x9, #8
    st1 {v7.s}[2], [X9]
    add x9, x9, #4
    
    // store column 3
    st1 {v8.4s}, [x9]
    add x9, x9, #16
    st1 {v9.4s}, [x9]
    add x9, x9, #16
    st1 {v10.4s}, [x9]
    add x9, x9, #16
    st1 {v11.2s}, [x9]
    add x9, x9, #8
    st1 {v11.s}[2], [X9]
    add x9, x9, #4
    
    // store column 4
    st1 {v12.4s}, [x9]
    add x9, x9, #16
    st1 {v13.4s}, [x9]
    add x9, x9, #16
    st1 {v14.4s}, [x9]
    add x9, x9, #16
    st1 {v15.2s}, [x9]
    add x9, x9, #8
    st1 {v15.s}[2], [X9]
    add x9, x9, #4

    // store column 5
    st1 {v16.4s}, [x9]
    add x9, x9, #16
    st1 {v17.4s}, [x9]
    add x9, x9, #16
    st1 {v18.4s}, [x9]
    add x9, x9, #16
    st1 {v19.2s}, [x9]
    add x9, x9, #8
    st1 {v19.s}[2], [X9]
    add x9, x9, #4
    
    // store column 6
    st1 {v20.4s}, [x9]
    add x9, x9, #16
    st1 {v21.4s}, [x9]
    add x9, x9, #16
    st1 {v22.4s}, [x9]
    add x9, x9, #16
    st1 {v23.2s}, [x9]
    add x9, x9, #8
    st1 {v23.s}[2], [X9]
    add x9, x9, #4


    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ret
