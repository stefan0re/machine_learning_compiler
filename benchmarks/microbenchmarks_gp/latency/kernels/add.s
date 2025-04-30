    .text
    .type add_latency, %function
    .global add_latency

add_latency:

    mov x1, #1
    mov x2, #1
    mov x3, #1
    mov x4, #1

    mov x5, #1
    mov x6, #1
    mov x7, #1
    mov x8, #1
    mov x9, #1
    mov x10, #1
    mov x11, #1
    mov x12, #1
    mov x13, #1
    mov x14, #1
    mov x15, #1
    mov x16, #1
    mov x17, #1

loop_0:
    sub x0, x0, #1

    .rept 100

    add x1, x16, x17
    add x2, x1, x17
    add x3, x2, x17
    add x4, x3, x17

    add x5, x4, x17
    add x6, x5, x17
    add x7, x6, x17
    add x8, x7, x17

    add x9, x8, x17
    add x10, x9, x17
    add x11, x10, x17
    add x12, x11, x17

    add x13, x12, x17
    add x14, x13, x17
    add x15, x14, x17
    add x16, x15, x17


    .endr


    cbnz x0, loop_0

    mov x0, #16*100

    ret


    .type add_throughput, %function
    .global add_throughput

add_throughput:

    mov x1, #1
    mov x2, #1
    mov x3, #1
    mov x4, #1

    mov x5, #1
    mov x6, #1
    mov x7, #1
    mov x8, #1
    mov x9, #1
    mov x10, #1
    mov x11, #1
    mov x12, #1
    mov x13, #1
    mov x14, #1
    mov x15, #1
    mov x16, #1
    mov x17, #1

loop_1:
    sub x0, x0, #1

    add x1, x16, x17
    add x2, x16, x17
    add x3, x16, x17
    add x4, x16, x17

    add x5, x16, x17
    add x6, x16, x17
    add x7, x16, x17
    add x8, x16, x17

    add x9, x16, x17
    add x10, x16, x17
    add x11, x16, x17
    add x12, x16, x17

    add x13, x16, x17
    add x14, x16, x17



    cbnz x0, loop_1

    mov x0, #14

    ret
