    .text
    .type copy_asm_0, %function
    .global copy_asm_0
 copy_asm_0:
    
    ldr w2, [x0, #0]      // w2 = a[0]
    str w2, [x1, #0]      // b[0] = w2

    ldr w3, [x0, #4]      // w3 = a[1]
    str w3, [x1, #4]      // b[1] = w3

    ldr w4, [x0, #8]      // w4 = a[2]
    str w4, [x1, #8]      // b[2] = w4

    ldr w5, [x0, #12]     // w5 = a[3]
    str w5, [x1, #12]     // b[3] = w5

    ldr w6, [x0, #16]     // w6 = a[4]
    str w6, [x1, #16]     // b[4] = w6

    ldr w7, [x0, #20]     // w7 = a[5]
    str w7, [x1, #20]     // b[5] = w7

    ldr w2, [x0, #24]     // reusing w2 = a[6]
    str w2, [x1, #24]     // b[6] = w2

    ret


    .text
    .type copy_asm_1, %function
    .global copy_asm_1
copy_asm_1:
    
    mov     x3, #0              // i = 0

loop_start:
    cmp     x3, x0              // compare i with n
    b.ge    loop_end            // if i >= n, break

    lsl     x4, x3, #2          // x4 = i * 4 (offset in bytes)

    ldr     w5, [x1, x4]        // w5 = a[i]
    str     w5, [x2, x4]        // b[i] = w5

    add     x3, x3, #1          // i++

    b       loop_start          // repeat

loop_end:
    ret
