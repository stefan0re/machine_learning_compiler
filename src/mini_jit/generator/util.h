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

    /**
     * @brief Get the two kernel sizes for the microkernels.
     *
     * @return used vector register count
     * @param i_m The number of rows in the matrix A.
     * @param i_n The number of columns in the matrix B.
     * @param o_kernel_1 The first kernel size.
     * @param o_kernel_2 The second kernel size
     */
    static int32_t get_kernel_sizes(int32_t i_m,
                                    int32_t i_n,
                                    int32_t o_kernel_1[2],
                                    int32_t o_kernel_2[2]);

    /**
     * @brief Generate microkernels.
     * @param i_kernel_sizes The kernel sizes.
     * @param i_used_vector_reg_count The number of used vector registers.
     */
    static void gen_microkernel(int32_t i_kernel_sizes[2],
                                int32_t i_used_vector_reg_count);

    /**
     * @brief Load C block for the given kernel sizes.
     * @param i_kernel_sizes The kernel sizes.
     */
    static void gen_c_load(int32_t i_kernel_sizes[2]);

    /**
     * @brief Store C block for the given kernel sizes.
     * @param i_kernel_sizes The kernel sizes.
     */
    static void gen_c_store(int32_t i_kernel_sizes[2]);
};
#endif
