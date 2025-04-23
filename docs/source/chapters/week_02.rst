Base Instructions
=================

Task 1: Copying Data
--------------------

Using only basic instructions, copying 7 values from a given array to another requires 7 
load and store operations. The register x0 holds the address of the first array, and the 
register x1 holds the address of the second array. Since addresses are 64-bit, x registers 
are used, and because the values are 32-bit, w registers are used. Each value has an offset 
of 4 bytes.

.. code-block:: text
    :linenos:
    
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

Using only basic instructions, copying n values from one given array to another 
requires n load and store operations. Therefore, a loop is needed. The registers 
and offsets are as described above. The structure is similar to the given for loop 
in the copy_c.c counterpart.

.. code-block:: text
    :linenos:

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

**Note:** The copy_driver.cpp file already includes tests for copy_asm_0 and copy_asm_1; therefore, separate unit tests have been omitted.

Task 2: Instruction Throughput and Latency
------------------------------------------

Throughput
``````````
In order to microbenchmark the :code:`ADD` shifted register, we write an assembly script containing the benchmark and a C++ driver executing the benchmark and taking the execution time.

In the benchmark file we first initialize all used register with :code:`MOV`. 
Then we execute the benchmark loop:

.. code-block:: text
    :linenos:

    loop:
        sub x0, x0, #1

        add x1, x1, x2, lsl #1
        add x3, x3, x4, lsr #1
        add x5, x5, x6, lsl #2
        add x7, x7, x8, asr #1

        add x9, x9, x10, lsl #1
        add x11, x11, x12, lsr #1
        add x13, x13, x14, lsl #2
        add x15, x15, x16, asr #1

        cbnz x0, loop

The loop is executed N times with N being passed in register x0, which is used as an iteration counter. First we subtract 1 from the counter and execute the :code:`ADD` shifted registers 8 times.
Afterwards we use :code:`cbnz` which jumps to the loop label if our counter is not zero.

This function is called by our C++ driver:

.. code-block:: C++

    start = std::chrono::high_resolution_clock::now();
    benchmark_add_shifted_registers(iterations);
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration<double>(end - start).count();
    throughput = (iterations / duration) * 8;  // 8 ops in one iter

For 10^9 iterations we get a execution duration of 0.570716 seconds and a throughput of 14.0175 GOPS.

For the :code:`MUL` operation, we use the following loop:

.. code-block:: text
    :linenos:

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

For 10^9 iterations we get a execution duration of 0.691266 seconds and a throughput of 11.5730 GOPS.


Latency
```````


We all worked on the tasks in equal parts.