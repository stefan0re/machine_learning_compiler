        .text
        .type matmul_16_6_1, %function
        .global matmul_16_6_1

matmul_16_6_1:

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // store C pointer
    mov x7, x2

    // adjust leading dimension
    lsl x4, x4, #2
    lsl x5, x5, #2

    // load C 
    ld1 {v0.4s-v3.4s}, [x2]
    add x2, x2, x5
    ld1 {v4.4s-v7.4s}, [x2]
    add x2, x2, x5
    ld1 {v8.4s-v11.4s}, [x2]
    add x2, x2, x5
    ld1 {v12.4s-v15.4s}, [x2]
    add x2, x2, x5
    ld1 {v16.4s-v19.4s}, [x2]
    add x2, x2, x5
    ld1 {v20.4s-v23.4s}, [x2]

    // load A
    ld1 {v24.4s - v27.4s}, [x0]

    ldr s28, [x1]
    add x1, x1, x4
    ldr s29, [x1]
    add x1, x1, x4

    ldr s30, [x1]
    add x1, x1, x4
    ldr s31, [x1]
    add x1, x1, x4

    fmla v0.4s, v24.4s, v28.s[0]
    fmla v1.4s, v25.4s, v28.s[0]
    fmla v2.4s, v26.4s, v28.s[0]
    fmla v3.4s, v27.4s, v28.s[0]

    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.4s, v25.4s, v29.s[0]
    fmla v6.4s, v26.4s, v29.s[0]
    fmla v7.4s, v27.4s, v29.s[0]

    ldr s28, [x1]
    add x1, x1, x4
    ldr s29, [x1]

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


    st1 {v0.4s-v3.4s}, [x7]
    add x7, x7, x5
    st1 {v4.4s-v7.4s}, [x7]
    add x7, x7, x5
    st1 {v8.4s-v11.4s}, [x7]
    add x7, x7, x5
    st1 {v12.4s-v15.4s}, [x7]
    add x7, x7, x5
    st1 {v16.4s-v19.4s}, [x7]
    add x7, x7, x5
    st1 {v20.4s-v23.4s}, [x7]


    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16


    ret

    .type matmul_16_6_64, %function
    .global matmul_16_6_64
matmul_16_6_64:

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // store pointers
    mov x7, x0
    mov x8, x1
    mov x9, x2

    // adjust leading dimension
    lsl x3, x3, #2
    lsl x4, x4, #2
    lsl x5, x5, #2

    mov x15, #5
    mul x15, x15, x4

    // K loop counter
    mov x10, #64

    // load C 
    ld1 {v0.4s-v3.4s}, [x9]
    add x9, x9, x5
    ld1 {v4.4s-v7.4s}, [x9]
    add x9, x9, x5
    ld1 {v8.4s-v11.4s}, [x9]
    add x9, x9, x5
    ld1 {v12.4s-v15.4s}, [x9]
    add x9, x9, x5
    ld1 {v16.4s-v19.4s}, [x9]
    add x9, x9, x5
    ld1 {v20.4s-v23.4s}, [x9]


k1_loop:

    sub x10, x10, #1

    // load A
    ld1 {v24.4s - v27.4s}, [x0]
    add x0, x0, x3

    ldr s28, [x1]
    add x1, x1, x4
    ldr s29, [x1]
    add x1, x1, x4

    ldr s30, [x1]
    add x1, x1, x4
    ldr s31, [x1]
    add x1, x1, x4

    fmla v0.4s, v24.4s, v28.s[0]
    fmla v1.4s, v25.4s, v28.s[0]
    fmla v2.4s, v26.4s, v28.s[0]
    fmla v3.4s, v27.4s, v28.s[0]

    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.4s, v25.4s, v29.s[0]
    fmla v6.4s, v26.4s, v29.s[0]
    fmla v7.4s, v27.4s, v29.s[0]

    ldr s28, [x1]
    add x1, x1, x4
    ldr s29, [x1]

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

    sub x1, x1, x15
    add x1, x1, #4

    cbnz x10, k1_loop

    mov x9, x2

    st1 {v0.4s-v3.4s}, [x9]
    add x9, x9, x5
    st1 {v4.4s-v7.4s}, [x9]
    add x9, x9, x5
    st1 {v8.4s-v11.4s}, [x9]
    add x9, x9, x5
    st1 {v12.4s-v15.4s}, [x9]
    add x9, x9, x5
    st1 {v16.4s-v19.4s}, [x9]
    add x9, x9, x5
    st1 {v20.4s-v23.4s}, [x9]


    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16


    ret


    .type matmul_64_6_64, %function
    .global matmul_64_6_64
matmul_64_6_64:

    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // store pointers
    mov x7, x0
    mov x8, x1
    mov x9, x2
    
    // adjust leading dimension
    lsl x3, x3, #2
    lsl x4, x4, #2
    lsl x5, x5, #2

    mov x15, #5
    mul x15, x15, x4

    // M loop counter
    mov x11, #4

m2_loop:
    sub x11, x11, #1

    // load C 
    ld1 {v0.4s-v3.4s}, [x9]
    add x9, x9, x5
    ld1 {v4.4s-v7.4s}, [x9]
    add x9, x9, x5
    ld1 {v8.4s-v11.4s}, [x9]
    add x9, x9, x5
    ld1 {v12.4s-v15.4s}, [x9]
    add x9, x9, x5
    ld1 {v16.4s-v19.4s}, [x9]
    add x9, x9, x5
    ld1 {v20.4s-v23.4s}, [x9]

    // K loop counter
    mov x10, #64


k2_loop:

    sub x10, x10, #1

    // load A
    ld1 {v24.4s - v27.4s}, [x7]
    add x7, x7, x3

    ldr s28, [x8]
    add x8, x8, x4
    ldr s29, [x8]
    add x8, x8, x4

    ldr s30, [x8]
    add x8, x8, x4
    ldr s31, [x8]
    add x8, x8, x4

    fmla v0.4s, v24.4s, v28.s[0]
    fmla v1.4s, v25.4s, v28.s[0]
    fmla v2.4s, v26.4s, v28.s[0]
    fmla v3.4s, v27.4s, v28.s[0]

    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.4s, v25.4s, v29.s[0]
    fmla v6.4s, v26.4s, v29.s[0]
    fmla v7.4s, v27.4s, v29.s[0]

    ldr s28, [x8]
    add x8, x8, x4
    ldr s29, [x8]

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

    sub x8, x8, x15

    add x8, x8, #4

    cbnz x10, k2_loop

    mov x9, x2

    st1 {v0.4s-v3.4s}, [x9]
    add x9, x9, x5
    st1 {v4.4s-v7.4s}, [x9]
    add x9, x9, x5
    st1 {v8.4s-v11.4s}, [x9]
    add x9, x9, x5
    st1 {v12.4s-v15.4s}, [x9]
    add x9, x9, x5
    st1 {v16.4s-v19.4s}, [x9]
    add x9, x9, x5
    st1 {v20.4s-v23.4s}, [x9]

    add x0, x0, #64
    add x2, x2, #64

    mov x7, x0
    mov x8, x1
    mov x9, x2

    cbnz x11, m2_loop


    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16

    ret


    .type matmul_64_48_64, %function
    .global matmul_64_48_64
matmul_64_48_64:

stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // store pointers
    mov x7, x0
    mov x15, x0
    mov x8, x1
    mov x9, x2

    // adjust leading dimension
    lsl x3, x3, #2
    lsl x4, x4, #2
    lsl x5, x5, #2

    // n loop offset
    mov x13, x4
    mov x14, #6
    mul x13, x13, x14

    mov x16, x5
    mov x14, #6
    mul x16, x16, x14

    mov x17, #5
    mul x17, x17, x4


    

    // N loop counter
    mov x12, #8

n3_loop:
    sub x12, x12, #1
    // M loop counter
    mov x11, #4

m3_loop:
    sub x11, x11, #1

    // load C 
    ld1 {v0.4s-v3.4s}, [x9]
    add x9, x9, x5
    ld1 {v4.4s-v7.4s}, [x9]
    add x9, x9, x5
    ld1 {v8.4s-v11.4s}, [x9]
    add x9, x9, x5
    ld1 {v12.4s-v15.4s}, [x9]
    add x9, x9, x5
    ld1 {v16.4s-v19.4s}, [x9]
    add x9, x9, x5
    ld1 {v20.4s-v23.4s}, [x9]

    // K loop counter
    mov x10, #64


k3_loop:

    sub x10, x10, #1

    // load A
    ld1 {v24.4s - v27.4s}, [x7]
    add x7, x7, x3

    ldr s28, [x8]
    add x8, x8, x4
    ldr s29, [x8]
    add x8, x8, x4

    ldr s30, [x8]
    add x8, x8, x4
    ldr s31, [x8]
    add x8, x8, x4

    fmla v0.4s, v24.4s, v28.s[0]
    fmla v1.4s, v25.4s, v28.s[0]
    fmla v2.4s, v26.4s, v28.s[0]
    fmla v3.4s, v27.4s, v28.s[0]

    fmla v4.4s, v24.4s, v29.s[0]
    fmla v5.4s, v25.4s, v29.s[0]
    fmla v6.4s, v26.4s, v29.s[0]
    fmla v7.4s, v27.4s, v29.s[0]

    ldr s28, [x8]
    add x8, x8, x4
    ldr s29, [x8]

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

    sub x8, x8, x17

    add x8, x8, #4

    cbnz x10, k3_loop

    mov x9, x2

    st1 {v0.4s-v3.4s}, [x9]
    add x9, x9, x5
    st1 {v4.4s-v7.4s}, [x9]
    add x9, x9, x5
    st1 {v8.4s-v11.4s}, [x9]
    add x9, x9, x5
    st1 {v12.4s-v15.4s}, [x9]
    add x9, x9, x5
    st1 {v16.4s-v19.4s}, [x9]
    add x9, x9, x5
    st1 {v20.4s-v23.4s}, [x9]

    add x0, x0, #64
    add x2, x2, #64

    mov x7, x0
    mov x8, x1
    mov x9, x2

    cbnz x11, m3_loop

    sub x2, x2, #64*4

    // set A
    mov x0, x15
    mov x7, x15

    // set B
    add x1, x1, x13
    mov x8, x1

    // set C
    add x2, x2, x16
    mov x9, x2
    cbnz x12, n3_loop


    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16


    ret

