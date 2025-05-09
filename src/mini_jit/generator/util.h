#ifndef MINI_JIT_GENERATOR_UTIL_H
#define MINI_JIT_GENERATOR_UTIL_H

#include <cstdint>
#include <string>

#include "../backend/Kernel.h"
#include "../instructions/instructions.h"

namespace mini_jit::generator {
    class Util;
}

class mini_jit::generator::Util {
   public:
    typedef enum : uint32_t {
        INPUT_ADDRESS_A_REG = mini_jit::instructions::InstGen::x0,
        INPUT_ADDRESS_B_REG = mini_jit::instructions::InstGen::x1,
        INPUT_ADDRESS_C_REG = mini_jit::instructions::InstGen::x2,

        WORKING_ADDRESS_A_REG = mini_jit::instructions::InstGen::x7,
        WORKING_ADDRESS_B_REG = mini_jit::instructions::InstGen::x8,
        WORKING_ADDRESS_C_REG = mini_jit::instructions::InstGen::x9,

        LEADING_DIM_A_REG = mini_jit::instructions::InstGen::x3,
        LEADING_DIM_B_REG = mini_jit::instructions::InstGen::x4,
        LEADING_DIM_C_REG = mini_jit::instructions::InstGen::x5,

        K_LOOP_COUNT_REG = mini_jit::instructions::InstGen::x10,
        M_LOOP_COUNT_REG = mini_jit::instructions::InstGen::x11,
        N_LOOP_COUNT_REG = mini_jit::instructions::InstGen::x12,

        HELP_REG_1 = mini_jit::instructions::InstGen::x13,
        HELP_REG_2 = mini_jit::instructions::InstGen::x14,
        HELP_REG_3 = mini_jit::instructions::InstGen::x15,
        HELP_REG_4 = mini_jit::instructions::InstGen::x16

    } user_reg_t;

    struct KernelSize {
        int M;
        int N;
    };

    struct KernelSizes {
        KernelSize kernel1;
        KernelSize kernel2;
    };

    /**
     * @brief Get the two kernel sizes for the microkernels.
     *
     * @param i_m The number of rows in the matrix A.
     * @param i_n The number of columns in the matrix B.
     * @param kernelsizes The size of each kernel.
     */
    static void get_kernel_sizes(int32_t i_m,
                                 int32_t i_n,
                                 KernelSizes kernelsizes);

    /**
     * @brief Generate microkernels.
     * @param kernel The kernel sizes.
     * @param i_used_vector_reg_count The number of used vector registers.
     */
    static void gen_microkernel(KernelSize kernelsize,
                                int32_t i_used_vector_reg_count);

    /**
     * @brief Load C block for the given kernel sizes.
     * @param kernel The kernel sizes.
     * @return used vector register count
     */
    static int32_t gen_c_load(KernelSize kernelsize);

    /**
     * @brief Store C block for the given kernel sizes.
     * @param kernel The kernel sizes.
     */
    static void gen_c_store(KernelSize kernelsize);
};
#endif
