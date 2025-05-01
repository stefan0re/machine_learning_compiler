NEON
=================

Task 1: Execution Throughput and Latency
----------------------------------------

Throughput
__________

This week we will familiarize ourselves with single instruction multiple data from Arm, using the NEON extension.
In this task we take a closer look at the FMLA and at the FMADD instruction. FMLA ist a typicall SIMD instruction, multipling to Vectors by element and accumulate the result on a third vector. 
FMADD, on the other hand is a scalar operation, multipling to scalars and adding the result on a third scalar, the result is written to another vector view of a vector register.

To measure the throughput of both instruction we looped around a lot of independent instructions.
This is our result for the two different FMLA instructions:

.. code-block:: text

    ---------------------------------
    Throughput FMLA 4S...
    Duration: 1.49625
    GFLOPS: 128.321
    ---------------------------------
    Throughput FMLA 2S...
    Duration: 1.44526
    GFLOPS: 66.4242
    ---------------------------------

If we assume a clook speed of 4.3 Ghz, a FMLA instruction should have a throughput of nearly 4 instructions per cycle.

The performance for the FMADD is shown here:


.. code-block:: text

    ---------------------------------
    Throughput FMADD...
      Duration: 1.3654
      GFLOPS: 30.7603
    ---------------------------------

Again if we devide the GFLOPS by the clock speed and by the FLOPS per instruction we get a little bit less the 4. So we assume again a throughput of 4 instructions per cycle.

Latency
_______

To see the latency of the FMLA instruction we run two different kernels, what has dependent instructions on the input and output vector register (SRC).
The other kernel has the dependency only on the accumulate register, so there is just a write after write dependency. (DST)



.. code-block:: text

    ---------------------------------
    Latency SRC FMLA 4S...
      Duration: 2.18838
      GFLOPS: 11.3326
    ---------------------------------
    Latency DST FMLA 4S...
      Duration: 2.18902
      GFLOPS: 11.3293
    ---------------------------------

As one can see the two kernels produce the same result. By using our throughput benchmark as a baseline we calculate that there is an instruction latency of 3 cycles for the FMLA instruction. We can use only 1 pipline instead of using 4 as suggested in the throughput part. An the we have two multiple our result with 3 to get to the full throughput. The multiple by 3 operation suggests that the instruction has a 3 cycle latency.
A funny side fact is that in our CI (Github Runner ARM) there is a difference between the SRC and DST variant.  Here the DST variant has double the throughput compared to the SRC variant.

Our benchmark code can be found `here <https://github.com/stefan0re/machine_learning_compiler/tree/main/benchmarks/microbenchmarks_neon>`_.

Task 2: Microkernels for Matrix-Multiplication
----------------------------------------------

Implementation with m=16, n=6, k=1
__________________________________

This implementation stands for one iteration of the inner k-loop.
Therefore one column vector (A) is multiplied to a row vector (B).
First, we load the complete target matrix values C completely into the vector 
register.

.. code-block:: text

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

Afterwords we load the A vector into a register.

.. code-block:: text

   // load A
   ld1 {v24.4s - v27.4s}, [x0]

Then we, load the row vector B into 4 registers (for two register we have to reload). 
For this, we employ the offset in ldb.

.. code-block:: text

    ldr s28, [x1]
    add x1, x1, x4
    ldr s29, [x1]
    add x1, x1, x4

    ldr s30, [x1]
    add x1, x1, x4
    ldr s31, [x1]
    add x1, x1, x4

Finally, we use a broadcast fmla operation and multiply the column vector A with one element of the row vector B.

.. code-block:: text

   fmla v0.4s, v24.4s, v28.s[0]
   fmla v1.4s, v25.4s, v28.s[0]
   fmla v2.4s, v26.4s, v28.s[0]
   fmla v3.4s, v27.4s, v28.s[0]

This, we continue for each element of B. When this process is done, we store the values of C back.

Implementation with m=16, n=6, k=64
___________________________________

For this task, we draw a loop around our existing code and introduce a k counter.
In each iteration, we load a new column of A and a new row of B but still use the same C Matrix. 

Implementation with m=64, n=6, k=64
___________________________________

For this task, we draw a loop around our existing code and introduce a m counter.
In each iteration of this loop, we load a new tile from c (16 rows lower). Thus, we have to adjust our pointer to A and C.

Implementation with m=64, n=48, k=64
____________________________________

Finally, we draw a loop around our existing code and introduce a n counter.
In each iteration of this loop, we have to move 6 elements further in the n dimension in B and C.

Our implementation can be found can be found in `matmul_1.s <https://github.com/stefan0re/machine_learning_compiler/tree/main/assembly_examples/task_3/kernels>`_.

Throughput
__________

We tested our throughput with corresponding randomly generated matrices. The results can be seen below and were stable across multiple executions.

.. code-block:: text

    ---------------------------------
    Testing matmul_16_6_1 ...

    Iterations: 150000000 times
    Duration: 0.976587 sec
    Throughput: 29.4905 GFLOPS

    ---------------------------------
    Testing matmul_16_6_64 ...

    Iterations: 10000000 times
    Duration: 1.01573 sec
    Throughput: 120.977 GFLOPS

    ---------------------------------
    Testing matmul_64_6_64 ...

    Iterations: 2000000 times
    Duration: 0.784322 sec
    Throughput: 125.336 GFLOPS

    ---------------------------------
    Testing matmul_64_48_64 ...

    Iterations: 250000 times
    Duration: 0.784602 sec
    Throughput: 125.292 GFLOPS


Our measurement environment can be found in `driver.s <https://github.com/stefan0re/machine_learning_compiler/tree/main/assembly_examples/task_3>`_.