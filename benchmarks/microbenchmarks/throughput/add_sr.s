.text
.type benchmark_add_shifted_registers, %function
.global benchmark_add_shifted_registers

benchmark_add_shifted_registers:
// initialize registers
    mov x1, #1
    mov x2, #2
    mov x3, #3
    mov x4, #4

    mov x5, #5
    mov x6, #6
    mov x7, #7
    mov x8, #8

    mov x9, #9
    mov x10, #10
    mov x11, #11
    mov x12, #12

    mov x13, #13
    mov x14, #14
    mov x15, #15
    mov x16, #16

//benchmark loop - 8 ops per iteration
loop:
    sub x0, x0, #1

    add x1, x1, x2, lsl, #1
    add x3, x3, x4, lsl, #1
    add x5, x5, x6, lsl, #1
    add x7, x7, x8, lsl, #1

    add x9, x9, x10, lsl, #1
    add x11, x11, x12, lsl, #1
    add x13, x13, x14, lsl, #1
    add x15, x15, x16, lsl, #1

    cbnz x0, loop

    ret


