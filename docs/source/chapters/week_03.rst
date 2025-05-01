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