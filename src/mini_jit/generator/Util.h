#ifndef MINI_JIT_GENERATOR_UTIL_H
#define MINI_JIT_GENERATOR_UTIL_H

#include <cstdint>
#include <string>
#include <vector>

#include "../backend/Kernel.h"
#include "../instructions/instructions.h"

namespace mini_jit::generator {

    class Util {
       private:
        static mini_jit::backend::Kernel m_kernel;

       public:
        inline static constexpr mini_jit::instructions::InstGen::gpr_t INPUT_ADDRESS_A_REG = mini_jit::instructions::InstGen::x0;
        inline static constexpr mini_jit::instructions::InstGen::gpr_t INPUT_ADDRESS_B_REG = mini_jit::instructions::InstGen::x1;
        inline static constexpr mini_jit::instructions::InstGen::gpr_t INPUT_ADDRESS_C_REG = mini_jit::instructions::InstGen::x2;

        inline static constexpr mini_jit::instructions::InstGen::gpr_t WORKING_ADDRESS_A_REG = mini_jit::instructions::InstGen::x7;
        inline static constexpr mini_jit::instructions::InstGen::gpr_t WORKING_ADDRESS_B_REG = mini_jit::instructions::InstGen::x8;
        inline static constexpr mini_jit::instructions::InstGen::gpr_t WORKING_ADDRESS_C_REG = mini_jit::instructions::InstGen::x9;

        inline static constexpr mini_jit::instructions::InstGen::gpr_t LEADING_DIM_A_REG = mini_jit::instructions::InstGen::x3;
        inline static constexpr mini_jit::instructions::InstGen::gpr_t LEADING_DIM_B_REG = mini_jit::instructions::InstGen::x4;
        inline static constexpr mini_jit::instructions::InstGen::gpr_t LEADING_DIM_C_REG = mini_jit::instructions::InstGen::x5;

        inline static constexpr mini_jit::instructions::InstGen::gpr_t K_LOOP_COUNT_REG = mini_jit::instructions::InstGen::x10;
        inline static constexpr mini_jit::instructions::InstGen::gpr_t M_LOOP_COUNT_REG = mini_jit::instructions::InstGen::x11;
        inline static constexpr mini_jit::instructions::InstGen::gpr_t N_LOOP_COUNT_REG = mini_jit::instructions::InstGen::x12;

        inline static constexpr mini_jit::instructions::InstGen::gpr_t HELP_REG_1 = mini_jit::instructions::InstGen::x13;
        inline static constexpr mini_jit::instructions::InstGen::gpr_t HELP_REG_2 = mini_jit::instructions::InstGen::x14;
        inline static constexpr mini_jit::instructions::InstGen::gpr_t HELP_REG_3 = mini_jit::instructions::InstGen::x15;
        inline static constexpr mini_jit::instructions::InstGen::gpr_t HELP_REG_4 = mini_jit::instructions::InstGen::x16;

        struct KernelSize {
            int M;
            int N;
        };

        struct KernelSizes {
            KernelSize kernel1;
            KernelSize kernel2;
            KernelSize kernel3;
            KernelSize kernel4;
        };

        /**
         * @brief Get the four subarea sizes for a matrix C.
         *
         * @param i_m vertical dimensions of C.
         * @param i_n horizontal dimensions of C.
         * @param kernelsizes The size of each area.
         */
        static void get_area_sizes(int32_t m,
                                   int32_t n,
                                   std::vector<KernelSize>& work_areas);

        /**
         * @brief Get the four kernel sizes for the microkernels.
         *
         * @param i_m The number of rows in the matrix A.
         * @param i_n The number of columns in the matrix B.
         * @param kernelsizes The size of each kernel.
         */
        static void get_kernel_sizes(int32_t m,
                                     int32_t n,
                                     KernelSizes& kernelsizes);

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
        static int32_t gen_matrix_load(mini_jit::backend::Kernel& m_kernel, KernelSize kernelsize, mini_jit::instructions::InstGen::gpr_t pointer_register, uint32_t leading_dimension);

        /**
         * @brief Store C block for the given kernel sizes.
         * @param kernel The kernel sizes.
         */
        static void gen_matrix_store(mini_jit::backend::Kernel& m_kernel, KernelSize kernelsize, mini_jit::instructions::InstGen::gpr_t pointer_register, uint32_t leading_dimension);
    };
}  // namespace mini_jit::generator
#endif
