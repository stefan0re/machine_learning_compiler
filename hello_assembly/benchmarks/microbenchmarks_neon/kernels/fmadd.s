    .text
    .type fmadd_throughput, %function
    .global fmadd_throughput

fmadd_throughput:

    // save callee-saved registers
    stp  d8,  d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!

    // zero neon register
    eor v0.16b, v0.16b, v0.16b
    eor v1.16b, v1.16b, v1.16b
    eor v2.16b, v2.16b, v2.16b
    eor v3.16b, v3.16b, v3.16b

    eor v4.16b, v4.16b, v4.16b
    eor v5.16b, v5.16b, v5.16b
    eor v6.16b, v6.16b, v6.16b
    eor v7.16b, v7.16b, v7.16b

    eor v8.16b, v8.16b, v8.16b
    eor v9.16b, v9.16b, v9.16b
    eor v10.16b, v10.16b, v10.16b
    eor v11.16b, v11.16b, v11.16b

    eor v12.16b, v12.16b, v12.16b
    eor v13.16b, v13.16b, v13.16b
    eor v14.16b, v14.16b, v14.16b
    eor v15.16b, v15.16b, v15.16b

    eor v16.16b, v16.16b, v16.16b
    eor v17.16b, v17.16b, v17.16b
    eor v18.16b, v18.16b, v18.16b
    eor v19.16b, v19.16b, v19.16b

    eor v20.16b, v20.16b, v20.16b
    eor v21.16b, v21.16b, v21.16b
    eor v22.16b, v22.16b, v22.16b
    eor v23.16b, v23.16b, v23.16b

    eor v24.16b, v24.16b, v24.16b
    eor v25.16b, v25.16b, v25.16b
    eor v26.16b, v26.16b, v26.16b
    eor v27.16b, v27.16b, v27.16b

    eor v28.16b, v28.16b, v28.16b
    eor v29.16b, v29.16b, v29.16b
    eor v30.16b, v30.16b, v30.16b
    eor v31.16b, v31.16b, v31.16b
    


loop_fmadd_t:
    sub x0, x0, #1

    .rept 50

    fmadd s0, s29, s30, s31
    fmadd s1, s29, s30, s31
    fmadd s2, s29, s30, s31
    fmadd s3, s29, s30, s31

    fmadd s4, s29, s30, s31
    fmadd s5, s29, s30, s31
    fmadd s6, s29, s30, s31
    fmadd s7, s29, s30, s31

    fmadd s8, s29, s30, s31
    fmadd s9, s29, s30, s31
    fmadd s10, s29, s30, s31
    fmadd s11, s29, s30, s31

    fmadd s12, s29, s30, s31
    fmadd s13, s29, s30, s31
    fmadd s14, s29, s30, s31
    fmadd s15, s29, s30, s31

    fmadd s16, s29, s30, s31
    fmadd s17, s29, s30, s31
    fmadd s18, s29, s30, s31
    fmadd s19, s29, s30, s31

    fmadd s20, s29, s30, s31
    fmadd s21, s29, s30, s31
    fmadd s22, s29, s30, s31
    fmadd s23, s29, s30, s31

    fmadd s24, s29, s30, s31
    fmadd s25, s29, s30, s31
    fmadd s26, s29, s30, s31
    fmadd s27, s29, s30, s31

    .endr

    cbnz x0, loop_fmadd_t

    mov x0, #2*28*50

    // restore callee-saved registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp  d8,  d9, [sp], #16


    ret
