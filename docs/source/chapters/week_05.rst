GEMM Code-Generation
====================

The solutions to this second task can be found in `this directory <https://github.com/stefan0re/machine_learning_compiler/tree/main/src/mini_jit/generator>`_.

GEMM
____

Kernel blocking
+++++++++++++++

In order to be able to process all matrix sizes, we first need to figure out the sizes of the sub-matrices and kernels. Thus, we implemented a algorithm that return these kernel sizes based on the dimensions of the :math:`C` matrix. 

.. image:: ../_static/matrix_areas.png
    :alt: :math:`C` Matrix areas

The first thing to do is to make sure that there are enough vector registers for loading A and B columns/rows and for holding the C matrix.
We have decided to use a maximum blocking of 16 values in the M dimension (line 32), because they worked in the previous examples and then iteratively approach the use of all 32 vector registers. This can be seen in lines 15-17.
Derived from the “big” kernel, only the edge column is then created on the right-hand side, which will probably get a less good N (line 28).

.. code-block:: C++
    :linenos:

    void mini_jit::generator::Util::get_kernel_sizes_brgemm(int32_t m,
                                                            int32_t n,
                                                            mini_jit::generator::Util::KernelSize &kernelsize_big,
                                                            mini_jit::generator::Util::KernelSize &kernelsize_small,
                                                            int32_t &i_used_vector_reg_count_big,
                                                            int32_t &i_used_vector_reg_count_small) {
        int32_t max_n_blocking = 30;
        int32_t m_blocks = 0;
        if (m > 12) {
            m_blocks = 4;
        } else {
            m_blocks = (m + 3) / 4;  // up_div
        }

        while (m_blocks * max_n_blocking + m_blocks + 1 > 32) {
            max_n_blocking--;
        }

        if (max_n_blocking > n) {
            max_n_blocking = n;
        }

        kernelsize_big.M32 = (m > 15) ? 16 : m;
        kernelsize_big.N = max_n_blocking;
        i_used_vector_reg_count_big = m_blocks * max_n_blocking;

        kernelsize_small.M = (m > 15) ? 16 : m;
        kernelsize_small.N = n % max_n_blocking;

        i_used_vector_reg_count_small = m_blocks * kernelsize_small.N;
    }


Generator
+++++++++

The JIT-generator function for the GEMM is pretty much strate forward after this blocking.
The accumulation on a C submatrix is first made in a K loop.
For this we wrote a microkernel function that only handles the compute.
After this block we go to the next block in M direction until the whole M dimension is done.
The M loop is made around each nano-kernel, except for the last edge case.
The addresses for A and C must be recalculated. The B address only needs to be reset.
Around this is then looped again in N in order to calculate the sub-matrices up to the last edge case, which is then processed separately.

The full implementation can be seen in this `here <https://github.com/stefan0re/machine_learning_compiler/blob/main/src/mini_jit/generator/Brgemm.cpp>`_, and here are `tests <https://github.com/stefan0re/machine_learning_compiler/blob/main/test/mini_jit/test_gemm.cpp>`_ for all sizes.
In our test we have to test cases one with the leading dimensions equal to the M K and M, and another one with random bigger leading dimensions.

Performance
+++++++++++

To measure all possible cases and let Edward sweat a little, we did a GEMM sweep over M,N = 1-64 and K = {1,16,32,64,128}.
We have written the results of our sweep to a CSV file that can be analyzed using the `visualization tool <http://scalable.uni-jena.de/opt/gemm/>`_ from the lecture. 
`GEMM-CSV-Download <../_static/m4_gemm.csv>`__

.. image:: ../_static/vis_gemm.png
    :alt: Plot of GEMM performance

The mean performance of all GEMM kernels is 76.988 GFLOPS.
Our best kernels have a performance of around 127 GFLOPS.


Batch-Reduce GEMM
_________________

A batch reduce includes a further K dimension in the kernel, so it ensures that not only one loop is looped around the inner microkernel, but a second one is added.
The second K dimension does not have to be directly after the first one in the memory, because you can use BR_strides as an offset to the next K dimension address. 
Therefore, we added if statements to the existing GEMM code to handle the second K dimension.
After each inner K loop, the address of the next outer K is calculated with the runtime parameters :code:`br_stride_a` and :code:`br_stride_b` which are in a seperate generall purpose register.

Again as with our regular GEMM we have written tests to verify the correctness of the code. (`Link <https://github.com/stefan0re/machine_learning_compiler/blob/main/test/mini_jit/test_brgemm.cpp>`_).

Again our results can be seen in a CSV file that can be analyzed using the visualization tool from the lecture.
Since we always use the same batch reduce size of 16, we have simply created the same type of CSV again, knowing that a batch size is used.
`BRGEMM-CSV-Download <../_static/m4_brgemm.csv>`__


If you like to try your own settings, you can use the program :code:`check_brgemm` which you can find in the build/bin directory after building our project.
An example command to run it is:

.. code-block:: bash

    ./build/bin/check_brgemm 64 64 64 2