Unary Operations
================

Task 1: Unary Primitives
------------------------

The solutions to this second task can be found in `this directory <https://github.com/stefan0re/machine_learning_compiler/tree/main/src/mini_jit/generator>`_.

In order to implement the unary primitives space efficiently, we first define a code frame that will call the corresponding functions for the primitive. This code frame firstly always defines the areas of the matrix. This is done by a function that operates just as the get_kernel_sizes function from last week. As soon as we have defined all matrix areas, the procedure call standard calls are added to the buffer. Afterwards we iterate through each area: first, we put the pointers to the matrices to the beginning of the respective area.

.. code-block:: C++
    :linenos:

    // Store pointers of A and B to x7, x8
    m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x7,
                                                        inst::InstGen::x0));
    m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x8,
                                                        inst::InstGen::x1));

    // add offset for working area
    m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7,
                                                    inst::InstGen::x7,
                                                    (int32_t)area.offset,
                                                    0));
    m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x8,
                                                    inst::InstGen::x8,
                                                    (int32_t)area.offset,
                                                    0));

Then, we initialize both the :math:`n` and :math:`m` loops, by setting the counters and check-pointing the amount of instructions in the buffer. Inside the M loop, we first load the :math:`A` Matrix. Then we call the function that adds all instructions for either the zero, relu or identity primitives. Finally, we store all values in the :math:`B` Matrix.

Afterwards, the pointers jump to the next logical location in context of the loop and we generate the final procedure call standard instructions.

For us this works as long both :math:`n` and :math:`m` are potencies of 2, sadly a not found bug keeps us from using any other dimensions.

Zero
____

By utilizing the movi instruction we move zero into each register.

.. code-block:: C++
    :linenos:

    int32_t Unary::gen_unary_zero(mini_jit::generator::Util::KernelSize kernelsize) {
            // count how many vectors are in use
            int32_t reg_count = 0;
            int32_t op_count = 0;
            // m_kernel.add_instr(0x4F030480);  // place 100

            // total number of elements needed to load
            int count = kernelsize.M;
            int quads = count / 4;
            int rem = count % 4;

            for (int j = 0; j < kernelsize.N; j++) {
                // for each row with each quad = (4s)
                for (int i = 0; i < quads; i++) {
                    m_kernel.add_instr(inst::InstGen::neon_movi_zero(static_cast<inst::InstGen::simd_fp_t>(reg_count++), true, false));
                    op_count++;
                }
            }

            for (int i = 0; i < rem; i++) {
                // load one element at a time (.s[N])
                m_kernel.add_instr(
                    inst::InstGen::neon_movi_zero(
                        static_cast<inst::InstGen::simd_fp_t>(reg_count), true, false));
                op_count++;
            }

            return reg_count;
        }

This implementation yields in the following underwhelming results in the benchmark:

.. code-block:: text
    :linenos:

    ---------------------------------
    Benchmarking Unary: Zero 
    Matrix dimensions of 64x64
    16

    Iterations:     2500000 times
    Duration:       0.996954 sec
    Throughput:     10.2713 GFLOPS

    ---------------------------------
    Benchmarking Unary: Zero 
    Matrix dimensions of 512x512
    16

    Iterations:     1000000 times
    Duration:       0.438079 sec
    Throughput:     9.34991 GFLOPS

ReLU
____

We implemented the ReLU by, firstly loading zero into the register v31 with movi and for each register holding an value of :math:`A` calling fmax.

.. code-block:: C++
    :linenos:

    int32_t Unary::gen_unary_relu(mini_jit::generator::Util::KernelSize kernelsize) {
        // count how many vectors are in use
        int32_t reg_count = 0;
        int32_t op_count = 0;

        // total number of elements needed to load
        int count = kernelsize.M;
        int quads = count / 4;
        int rem = count % 4;
        m_kernel.add_instr(inst::InstGen::neon_movi_zero(inst::InstGen::simd_fp_t::v31, true, false));
        op_count++;

        for (int j = 0; j < kernelsize.N; j++) {
            // for each row with each quad = (4s)
            for (int i = 0; i < quads; i++) {
                m_kernel.add_instr(inst::InstGen::neon_fmax_vector(static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                   static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                   inst::InstGen::simd_fp_t::v31,
                                                                   false));
                reg_count++;
                op_count++;
            }
        }

        for (int i = 0; i < rem; i++) {
            // load one element at a time (.s[N])
            m_kernel.add_instr(inst::InstGen::neon_fmax_vector(static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                               static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                               inst::InstGen::simd_fp_t::v31,
                                                               false));
            op_count++;
        }

        return reg_count;
    }

Again this implementation is not as optimized as wished as the benchmarking results show:

.. code-block:: text
    :linenos:

    ---------------------------------
    Benchmarking Unary: Relu 
    Matrix dimensions of 64x64

    Iterations:     2500000 times
    Duration:       0.957973 sec
    Throughput:     10.6892 GFLOPS

    ---------------------------------
    Benchmarking Unary: Relu 
    Matrix dimensions of 512x512

    Iterations:     100000 times
    Duration:       0.0384401 sec
    Throughput:     10.6555 GFLOPS

Identity
____

We implemented the identity matrix in the simplest possible way: by iterating through the input matrix A element by element and storing each value in B with the corresponding offset based on the size (assuming that A and B are square matrices).

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


We all worked on the tasks in equal parts.
This week's work is available under this commit on GitHub: 
