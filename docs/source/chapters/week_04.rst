Code Generation & Matrix-Multiplication
=======================================

Task 1: SIMD lanes & Accumulator Block Shapes
---------------------------------------------

The solutions to this first task can be found in `this directory <https://github.com/stefan0re/machine_learning_compiler/tree/main/hello_assembly/assembly_examples/neon>`_.

matmul_14_6_64
______________

For the first task, we have to implement a kernel with :math:`M=14`. This is different to the former implementations, as one column will not fully fill four vector registers. Therefore the loads and stores of the :math:`A` and :math:`C` matrices have to be adjusted. For each vector register we use separate load instructions. For example for the first column of :math:`C`:

.. code-block:: text
    :linenos:

    // load C 
    // load column 1
    ld1 {v0.4s}, [x9]
    add x9, x9, #16
    ld1 {v1.4s}, [x9]
    add x9, x9, #16
    ld1 {v2.4s}, [x9]
    add x9, x9, #16
    ld1 {v3.2s}, [x9]
    add x9, x9, #8

It is important to mind that only half of the last register is used and the pointer is only incremented by 8 bytes.

The throughput test yields the following result:

.. code-block:: text

    Testing matmul_14_6_64 ...
    Iterations:     10000000 times
    Duration:       1.00084 sec
    Throughput:     122.777 GFLOPS

This shows a small decrease, which is explainable through the increase of memory accesses.

matmul_15_6_64
______________

Similarly to :code:`matmul_14_6_64`, we have again columns that do not fully fill 4 vector registers. Additionally, the :code:`LD1` (and :code:`ST1`) instruction does not allow to load (only) 3 single words with one call. Thus, we have to use an additional :code:`LD1` call to load the 15'th element of the column into the third single-precision lane of the fourth vector register in use. This has to be implemented for both :math:`A` and :math:`C`. For example for the first column of :math:`C`:

.. code-block:: text
    :linenos:

    // load C 
    // load column 1
    ld1 {v0.4s}, [x9]
    add x9, x9, #16
    ld1 {v1.4s}, [x9]
    add x9, x9, #16
    ld1 {v2.4s}, [x9]
    add x9, x9, #16
    ld1 {v3.2s}, [x9]
    add x9, x9, #8
    ld1 {v3.s}[2], [X9]
    add x9, x9, #4

Again the pointer has to be increased for another 4 bytes.

The throughput test yields the following result:

.. code-block:: text

    Testing matmul_15_6_64 ...
    Iterations:     10000000 times
    Duration:       1.07793 sec
    Throughput:     113.996 GFLOPS

The throughput is again reduced by additional memory accesses.

matmul_64_64_64
_______________

In this exercise, we have to reshape our kernel, as :math:`N=64\text{ % }6\neq0`. Another restriction is that the whole accumulator block has to fit into the available register. Therefore,  bigger matrix sizes of :math:`16\times16` and :math:`16\times8`, which reduce the amount of memory communication, are not possible. Finally, the next smaller size is :math:`8\times8`, which we chose for our accumulator.

Compared to the implementation for :code:`matmul_64_48_64` we have in total 8 iterations in the :math:`M` loop. Furthermore, each column now only fills two vector registers, which inflates the amount of memory calls.

Our throughput test shows that this has an significant influence on the throughput:

.. code-block:: text

    Testing matmul_64_64_64 ...
    Iterations:     150000 times
    Duration:       1.31555 sec
    Throughput:     59.7796 GFLOPS


