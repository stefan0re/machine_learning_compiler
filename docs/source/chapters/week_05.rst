GEMM Code-Generation
====================

Task 1: Neon Batch-Reduce GEMM
------------------------------




Task 2: Code-Generation
-----------------------

The solutions to this second task can be found in `this directory <https://github.com/stefan0re/machine_learning_compiler/tree/main/src/mini_jit/generator>`_.

GEMM
____

get_kernel_sizes
++++++++++++++++

In order to be able to process all matrix sizes, we first need to figure out the sizes of the sub-matrices and kernels. Thus, we implemented a algorithm that return these kernel sizes based on the dimensions of the :math:`C` matrix. First, we split the matrix into 2-4 sub-matrices. We split in a dimension if it is not dividable by 4 (a kernel would not completely fill registers). If both dimension are not dividable by 4, :math:`C` looks as follows:

.. image:: ../_static/matrix_areas.png
    :alt: :math:`C` Matrix areas

After we have defined the areas, we define the kernel for each area. For each kernel, we iterate over all numbers between 1 and 16 for both :math:`m` and :math:`n` and calculate how many registers an :math:`A` column, :math:`B` row and the :math:`C` Matrix for this configuration of dimensions have to occupy. We filter all configuration of dimensions if this number of registers exceeds 32 (maximal number of registers) or if the respective kernel dimensions do not divide the dimensions of the area. If an kernel passed, we calculate the score which is a heuristic based on how square the kernel is, how much bigger :math:`m` is compared to :math:`n`, and how many registers will not be used by the kernel. 

.. code-block:: C++
    :linenos:

    // get used registers
    int32_t A_regs = (m_temp - (m_temp % 4)) / 4 + ((m_temp % 4 == 0) ? 0 : 1);
    int32_t B_regs = (n_temp - (n_temp % 4)) / 4 + ((n_temp % 4 == 0) ? 0 : 1);

    int32_t C_size = m_temp * n_temp;
    int32_t C_regs = (C_size - (C_size % 4)) / 4 + ((C_size % 4 == 0) ? 0 : 1);

    int32_t used_reg_space = A_regs + B_regs + C_regs;

    if (max_reg_space >= used_reg_space && (area.M % m_temp == 0 && area.N % n_temp == 0)) {
        // metric for how square the rectangle spanned by n_temp and m_temp is
        double squareness_deficit = fabs(((double)n_temp / (double)m_temp) - 1);

        // metrix for how much bigger m is compared to n
        double n_greater_m_deficit = (double)n_temp / (double)m_temp;

        // relative number of unused registers
        double registers_left = (max_reg_space - used_reg_space) / (double)max_reg_space;

        double score = w_sd * squareness_deficit + w_rl * registers_left + w_mn * n_greater_m_deficit;

        if (score < min_score) {
            min_score = score;
            best_m = m_temp;
            best_n = n_temp;
        }
    }

The kernel with the smallest score is chosen as the kernel for the respective area.

Batch-Reduce GEMM
_________________


We all worked on the tasks in equal parts.
This week's work is available under this commit on GitHub: 