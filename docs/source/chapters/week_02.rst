Base Instructions
=================

Task 1: Copying Data
--------------------

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