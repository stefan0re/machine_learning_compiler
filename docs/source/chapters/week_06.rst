Unary Operations
================

Task 1: Unary Primitives
------------------------

Unary primitives only work with an input matrix and an output matrix.
With the exception of the Transpose operation, all unary operations are performed element by element.
We have decided to use 4 different sub-generators and thus process transpose and identity independently of each other.

Zero Primitive
^^^^^^^^^^^^^^

The zero primitive basically does not need any loads in the registers, so we first set the registers v0-v3 to 0 with the eor instruction. 
Then we run over the M dimension and write the zeros to the corresponding places in the output tensor.
Finally we loop over the N dimension (columns) and tada, the values in the output matrix are 0.
Here is our performance for a few matrix sizes:

.. code-block:: text

    Running Unary ZERO Benchmark with: M = 50, N = 50
    Total Error: 0
    Performance: 109.604 GiB/s
    Iterations: 6996921
    Duration: 1.18908 seconds
    Running Unary ZERO Benchmark with: M = 64, N = 64
    Total Error: 0
    Performance: 123.136 GiB/s
    Iterations: 12370113
    Duration: 3.06575 seconds
    Running Unary ZERO Benchmark with: M = 512, N = 512
    Total Error: 0
    Performance: 129.489 GiB/s
    Iterations: 307847
    Duration: 4.64337 seconds
    Running Unary ZERO Benchmark with: M = 2048, N = 2048
    Total Error: 0
    Performance: 94.6051 GiB/s
    Iterations: 20712
    Duration: 6.8416 seconds

The performance of this kernel is quite acceptable, of course one have to keep in mind that it only writes and does not load anything
So the OPS calculation is perhaps a little misleading, here we include the the loads because the operation says we zero the input and write it to output.
However you do it, it is never completely correct :D.
The last case loses some performance, probably because the matrix is so large.

Identity Primitive
^^^^^^^^^^^^^^^^^^

The identity primitive is a bit more complex, as it has to load the input matrix and write it to the output matrix and be aware of the diffent leading dimension.
However, the same tactic is used as with the zero kernel, except that nothing has to be zeroed, so we use all vector registers for the copy operation.
Performance results can be seen here:

.. code-block:: text

    Running Unary IDENTITY Benchmark with: M = 50, N = 50
    Total Error: 0
    Performance: 115.572 GiB/s
    Iterations: 22857142
    Duration: 3.68384 seconds
    Running Unary IDENTITY Benchmark with: M = 64, N = 64
    Total Error: 0
    Performance: 130.021 GiB/s
    Iterations: 14999250
    Duration: 3.52053 seconds
    Running Unary IDENTITY Benchmark with: M = 512, N = 512
    Total Error: 0
    Performance: 107.17 GiB/s
    Iterations: 258340
    Duration: 4.70815 seconds
    Running Unary IDENTITY Benchmark with: M = 2048, N = 2048
    Total Error: 0
    Performance: 70.8844 GiB/s
    Iterations: 10681
    Duration: 4.70881 seconds

As you can see, the best performance is achieved with a 64x64 matrix. Here we come close to the peak value from the zero primitive again.


ReLU Primitive
^^^^^^^^^^^^^^

The ReLU implementation is most similar to that of Identity, but here the last vector register v31 cannot be used for copying as it is needed for the max comparison with 0.
Therefore we set it to 0 at the beginning with the eor instruction and perform a fmax comparison with the v31 register every time before we store a vector register with data from the input matrix into the ouput matrix.
The performance results are as follows:

.. code-block:: text

    Running Unary RELU Benchmark with: M = 50, N = 50
    Total Error: 0
    Performance: 112.422 GiB/s
    Iterations: 21819768
    Duration: 3.61516 seconds
    Running Unary RELU Benchmark with: M = 64, N = 64
    Total Error: 0
    Performance: 127.45 GiB/s
    Iterations: 15001500
    Duration: 3.59206 seconds
    Running Unary RELU Benchmark with: M = 512, N = 512
    Total Error: 0
    Performance: 105.272 GiB/s
    Iterations: 268575
    Duration: 4.98291 seconds
    Running Unary RELU Benchmark with: M = 2048, N = 2048
    Total Error: 0
    Performance: 70.9882 GiB/s
    Iterations: 10732
    Duration: 4.72438 seconds

As expected, the results are very similar to those from the identity implementation, as the fmax does not generate any real overhead.

Task 2: Transposition
---------------------
The neon 8 by 8 identity kernel can be found at: https://github.com/stefan0re/machine_learning_compiler/hello_assembly/assembly_examples/neon.
First, it consists of a set of loads:

.. code-block:: text
    :linenos:
    
    ld1     {v0.4s-v3.4s},   [x7], #64
    ld1     {v4.4s-v7.4s},   [x7], #64
    ld1     {v16.4s-v19.4s}, [x7], #64
    ld1     {v20.4s-v23.4s}, [x7], #64

Then, a set of trn1 and trn2 instructions for the 32-bit valued rows, divided into an upper and a lower part.

.. code-block:: text
    :linenos:

    // top half (shift 32-bit values)
    trn1    v24.4s, v0.4s, v2.4s    // row0
    trn1    v25.4s, v1.4s, v3.4s
    trn2    v26.4s, v0.4s, v2.4s    // row1
    trn2    v27.4s, v1.4s, v3.4s
    trn1    v28.4s, v4.4s, v6.4s    // row2
    trn1    v29.4s, v5.4s, v7.4s
    trn2    v30.4s, v4.4s, v6.4s    // row3
    trn2    v31.4s, v5.4s, v7.4s
    
    // bottom half (shift 32-bit values)
    trn1    v0.4s, v16.4s, v18.4s   // row4
    trn1    v1.4s, v17.4s, v19.4s
    trn2    v2.4s, v16.4s, v18.4s   // row5
    trn2    v3.4s, v17.4s, v19.4s
    trn1    v4.4s, v20.4s, v22.4s   // row6
    trn1    v5.4s, v21.4s, v23.4s
    trn2    v6.4s, v20.4s, v22.4s   // row7
    trn2    v7.4s, v21.4s, v23.4s

Followed by another trn1 and trn2 block for the 64-bit blocks, also divided into an upper and a lower part.
Note: Because there were not enough registers, the upper part is stored and the registers are reused for the lower part.

.. code-block:: text
    :linenos:

    // save to reuse registers 
    st1     {v16.4s-v19.4s}, [x8], #64
    st1     {v20.4s-v23.4s}, [x8], #64

    // bottom half (shift 64-bit values)
    trn1    v16.2d, v25.2d, v29.2d  // row4a
    trn1    v17.2d, v1.2d, v5.2d    // row4b
    trn1    v18.2d, v27.2d, v31.2d  // row5a
    trn1    v19.2d, v3.2d, v7.2d    // row5b
    trn2    v20.2d, v25.2d, v29.2d  // row4a
    trn2    v21.2d, v1.2d, v5.2d    // row4b
    trn2    v22.2d, v27.2d, v31.2d  // row5a
    trn2    v23.2d, v3.2d, v7.2d    // row5b

    // store B
    st1     {v16.4s-v19.4s}, [x8], #64
    st1     {v20.4s-v23.4s}, [x8]

This implementation is optimized:

.. code-block:: text
    :linenos:

    ---------------------------------
    Testing transpose_8_8
    
    Iterations:     300000000 times
    Duration:       1.22983 sec
    Throughput:     11.221 GFLOPS



Learning from this example, we then set to work on the generator for transpositions.


Transpose Primitive
^^^^^^^^^^^^^^^^^^^

In order not to lose track of the rows and columns, we have decided to write a fixed microkernel for 4x4 transpositions.
This works like the assembly code described above.
Therefore we loop as much as we can until we reach the limits in M and N where the kernel does not fit anymore because :code:`m_rest == 3` or :code:`n_rest == 2`.
For these cases we have the microkernel gen_transpose_micro_reminder, which work without tricky TRN or ZIP functions and uses the advantage that at most 12 values have to be transposed since either M or N is less than 4.
This kernel therefore loads each individual value to be transposed into a separate vector register and then stores the values at the correct position in the transposed output matrix.
So thank God for 32 vector registers, plenty of room for each value to stretch out and live the good life!
Here is the function that handles the edge cases:

.. code-block:: C++

    void Unary::gen_transpose_micro_reminder(uint32_t i_m,
                                            uint32_t i_n) {
        // load each value to seperate register
        int32_t v_reg_count = 0;
        for (size_t l_n = 0; l_n < i_n; l_n++) {
            for (size_t l_m = 0; l_m < i_m; l_m++) {
                m_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(v_reg_count++),
                                                        inst::InstGen::x0,
                                                        4,
                                                        inst::InstGen::arr_spec_t::s));
            }
            // set to next column
            m_kernel.add_instr(inst::InstGen::base_sub_imm(inst::InstGen::x0, inst::InstGen::x0, 4 * i_m, 0));
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x0,
                                                                        inst::InstGen::x0,
                                                                        inst::InstGen::x2,
                                                                        0,
                                                                        0));
        }
        v_reg_count = 0;
        // store values from seperate register
        for (size_t l_m = 0; l_m < i_m; l_m++) {
            for (size_t l_n = 0; l_n < i_n; l_n++) {
                m_kernel.add_instr(inst::InstGen::neon_str(static_cast<inst::InstGen::simd_fp_t>(v_reg_count),
                                                        inst::InstGen::x1,
                                                        4,
                                                        inst::InstGen::arr_spec_t::s));
                v_reg_count += i_m;
            }
            m_kernel.add_instr(inst::InstGen::base_sub_imm(inst::InstGen::x1, inst::InstGen::x1, 4 * i_n, 0));
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x1,
                                                                        inst::InstGen::x1,
                                                                        inst::InstGen::x3,
                                                                        0,
                                                                        0));
            v_reg_count -= i_m * i_n;
            v_reg_count += 1;
        }
    }

As with the GEMM, the transpose kernels are processed first in the M and then in the N direction.
Here are the performance results:

.. code-block:: text

    Running Unary TRANS Benchmark with: M = 50, N = 50
    Total Error: 0
    Performance: 65.3051 GiB/s
    Iterations: 15285845
    Duration: 4.35986 seconds
    Running Unary TRANS Benchmark with: M = 64, N = 64
    Total Error: 0
    Performance: 83.9639 GiB/s
    Iterations: 4411699
    Duration: 1.60348 seconds
    Running Unary TRANS Benchmark with: M = 512, N = 512
    Total Error: 0
    Performance: 4.1031 GiB/s
    Iterations: 11059
    Duration: 5.26421 seconds
    Running Unary TRANS Benchmark with: M = 2048, N = 2048
    Total Error: 0
    Performance: 3.50702 GiB/s
    Iterations: 546
    Duration: 4.86525 seconds

Our implementation seems to work quite well for smaller matrices, but unfortunately larger matrices the performance crashes.
We assume that the transposition simply generates out of control memory accesses that with larger matrices the memory subsystem no longer knows what is going on.
