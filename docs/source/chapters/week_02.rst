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

Now we want to look at the latency of the :code:`ADD` and :code:`MUL` instruction, for this we have to create dependencies in the assembly code in order to execute one instruction after the other sequentially.
To do this, we add a read after write dependency on a register so that no parallelism can be used in the processor core.

We loop over such dependencies in the assembly code:

.. code-block:: text
    :linenos:

    add x1, x16, x17
    add x2, x1, x17
    add x3, x2, x17
    add x4, x3, x17

As one can see, the instruction in line 2 depends on the completion of the instruction in line 1.

Again we loop over this code and measure the time of multiple iterations, with the same C++ driver as before.

Here we can measured a throughput of 4.3 GOPS for the :code:`ADD` instruction.
If we assume that the processor has an approximate clock frequency of 4.3 Ghz, we can say that the latency of the :code:`ADD` instruction is 1 instruction per cycle.
This would also be consistent with other ARM microarchitectures where the :code:`ADD` instruction also almost always has a latency of 1.
This also explains our assumption about the processor frequency.

We did the same again for the :code:`MUL` instruction and got a throughput with the read after write dependencies of 1.4 GOPS.
If we multiply this result by three, we come close to our assumed processor speed.
Therefore we conclude that the MUL instruction has a latency of 3 clock cycles. 

We all worked on the tasks in equal parts.