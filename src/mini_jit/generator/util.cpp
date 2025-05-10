#include "util.h"

#include "../instructions/instructions.h"

using mini_jit::instructions::InstGen;

namespace mini_jit::generator {

    mini_jit::backend::Kernel Util::m_kernel;

    void Util::get_kernel_sizes(int32_t i_m, int32_t i_n, Util::KernelSizes kernelsizes) {
    }

    void Util::gen_microkernel(Util::KernelSize kernelsize, int32_t i_used_vector_reg_count) {
        // shift leading dimensions to 4 bytes
        m_kernel.add_instr(InstGen::base_lsl_imm(InstGen::x3, InstGen::x3, 2));
        m_kernel.add_instr(InstGen::base_lsl_imm(InstGen::x4, InstGen::x4, 2));
        m_kernel.add_instr(InstGen::base_lsl_imm(InstGen::x5, InstGen::x5, 2));

        // set register to reset B in the K loop:
        m_kernel.add_instr(InstGen::base_mov_imm(InstGen::x15,
                                                 5,
                                                 0));
        m_kernel.add_instr(InstGen::base_mul_reg(InstGen::x15,
                                                 InstGen::x15,
                                                 InstGen::x4));
        // set K loop register
        m_kernel.add_instr(InstGen::base_mov_imm(InstGen::x10,
                                                 0,
                                                 0));

        // start k loop remember instruction count
        size_t k_loop_count = m_kernel.get_size();
        // sub k loop counter
        m_kernel.add_instr(InstGen::base_sub_imm(InstGen::x10,
                                                 InstGen::x10,
                                                 1,
                                                 0));

        m_kernel.write("debug_gen_microkernel.bin");
    }

    // I assume that get_kernel_size only return valid kernels, so there must be enough registers
    // to fit each value in and that the working registers are aligned already.
    int32_t Util::gen_c_load(Util::KernelSize kernelsize) {
        int remaining_elements = kernelsize.M * kernelsize.N;
        int i = 0;
        int vector_count_case = 0;

        while (remaining_elements > 0) {
            if (remaining_elements >= 4) {
                remaining_elements -= 4;
                vector_count_case = 4;
            } else if (remaining_elements >= 3) {
                remaining_elements -= 3;
                vector_count_case = 3;
            } else if (remaining_elements >= 2) {
                remaining_elements -= 2;
                vector_count_case = 2;
            } else if (remaining_elements >= 1) {
                remaining_elements -= 1;
                vector_count_case = 1;
            }

            m_kernel.add_instr(InstGen::neon_ld1_no_offset(static_cast<InstGen::simd_fp_t>(i),
                                                           Util::WORKING_ADDRESS_C_REG,
                                                           static_cast<InstGen::vector_count_t>(vector_count_case)));
            i++;

            m_kernel.add_instr(InstGen::base_add_imm(Util::WORKING_ADDRESS_C_REG,
                                                     Util::WORKING_ADDRESS_C_REG,
                                                     vector_count_case,
                                                     0));
        }

        m_kernel.write("debug_load_C.bin");

        return i;
    }

    void Util::gen_c_store(Util::KernelSize kernelsize) {
        int remaining_elements = kernelsize.M * kernelsize.N;
        int i = 0;
        int vector_count_case = 0;

        while (remaining_elements > 0) {
            if (remaining_elements >= 4) {
                remaining_elements -= 4;
                vector_count_case = 4;
            } else if (remaining_elements >= 3) {
                remaining_elements -= 3;
                vector_count_case = 3;
            } else if (remaining_elements >= 2) {
                remaining_elements -= 2;
                vector_count_case = 2;
            } else if (remaining_elements >= 1) {
                remaining_elements -= 1;
                vector_count_case = 1;
            }

            m_kernel.add_instr(InstGen::neon_st1_no_offset(static_cast<InstGen::simd_fp_t>(i),
                                                           Util::WORKING_ADDRESS_C_REG,
                                                           static_cast<InstGen::vector_count_t>(vector_count_case)));
            i++;
            m_kernel.add_instr(InstGen::base_add_imm(Util::WORKING_ADDRESS_C_REG,
                                                     Util::WORKING_ADDRESS_C_REG,
                                                     vector_count_case,
                                                     0));
        }

        m_kernel.write("debug_store_C.bin");
    }
}  // namespace mini_jit::generator
