.text
.type benchmark_mul, %function
.global benchmark_mul

benchmark_mul:
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

    mul x1, x1, x2
    mul x3, x3, x4
    mul x5, x5, x6
    mul x7, x7, x8

    mul x9, x9, x10
    mul x11, x11, x12
    mul x13, x13, x14
    mul x15, x15, x16

    cbnz x0, loop

    ret


