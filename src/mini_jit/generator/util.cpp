#include <Kernel.h>
#include <utils.h>

#include "src/mini_jit/instructions/instructions.h"

namespace mini_jit::generator {

    void Util::get_kernel_sizes(int32_t i_m, int32_t i_n, Util::KernelSizes kernelsizes) {
        // Your implementation
    }

    void Util::gen_microkernel(Util::KernelSize kernelsize, int32_t i_used_vector_reg_count) {
        // Your implementation
    }

    // I assume that get_kernel_size only return valid kernels, so there must be enough registers
    // to fit each value in and that the working registers are aligned already.
    int32_t Util::gen_c_load(Util::KernelSize kernelsize) {
        int remaining_elements = kernelsize.M * kernelsize.N;
        int i = 0;
        int vector_count_case = 0;

        wihle(remaining_elements >= 0) {
            if (remaining_elements - 4 >= 0) {
                remaining_elements -= 4;
                vector_count_case = 4;
            }
            elif (remaining_elements - 3 >= 0) {
                remaining_elements -= 3;
                vector_count_case = 3;
            }
            elif (remaining_elements - 2 >= 0) {
                remaining_elements -= 2;
                vector_count_case = 2;
            }
            elif (remaining_elements - 1 >= 0) {
                remaining_elements -= 1;
                vector_count_case = 1;
            }

            m_kernel.add_instr(InstGen::neon_ld1_no_offset(static_cast<inst::InstGen::simd_fp_t>(i),
                                                           Util::user_reg_t::WORKING_ADDRESS_C_REG,
                                                           static_cast<inst::InstGen::vector_count_t>(vector_count_case)));
            i++;
            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x9,
                                                           inst::InstGen::x9,
                                                           vector_count_case,
                                                           0));
        }

        return i;
    }

    void Util::gen_c_store(Util::KernelSize kernelsize) {
        int remaining_elements = kernelsize.M * kernelsize.N;
        int i = 0;
        int vector_count_case = 0;

        wihle(remaining_elements >= 0) {
            if (remaining_elements - 4 >= 0) {
                remaining_elements -= 4;
                vector_count_case = 4;
            }
            elif (remaining_elements - 3 >= 0) {
                remaining_elements -= 3;
                vector_count_case = 3;
            }
            elif (remaining_elements - 2 >= 0) {
                remaining_elements -= 2;
                vector_count_case = 2;
            }
            elif (remaining_elements - 1 >= 0) {
                remaining_elements -= 1;
                vector_count_case = 1;
            }

            m_kernel.add_instr(InstGen::neon_st1_no_offset(static_cast<inst::InstGen::simd_fp_t>(i),
                                                           Util::user_reg_t::WORKING_ADDRESS_C_REG,
                                                           static_cast<inst::InstGen::vector_count_t>(vector_count_case)));
            i++;
            m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x9,
                                                           inst::InstGen::x9,
                                                           vector_count_case,
                                                           0));
        }
    }
}  // namespace mini_jit::generator
