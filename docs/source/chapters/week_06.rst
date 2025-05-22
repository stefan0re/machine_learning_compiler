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

Task 2: Transposition
---------------------


We all worked on the tasks in equal parts.
This week's work is available under this commit on GitHub: 