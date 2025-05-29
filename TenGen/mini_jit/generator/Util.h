#ifndef TENGEN_MINI_JIT_GENERATOR_UTIL_H
#define TENGEN_MINI_JIT_GENERATOR_UTIL_H

#include <cfloat>
#include <cstdint>
#include <string>
#include <vector>

#include "TenGen/mini_jit/backend/Kernel.h"
#include "TenGen/mini_jit/instructions/Encoding.h"
#include "TenGen/types/Structs.h"
#include "TenGen/types/Types.h"

using namespace TenGen::Types;
using namespace TenGen::Structs;
using namespace TenGen::MiniJit::Instructions::Encoding;
using Kernel = TenGen::MiniJit::Backend::Kernel;

namespace TenGen::MiniJit::Generator::Util {
    /**
     * @brief Get the two kernel sizes for the microkernels and the loads/stores for the microkernel BRGEMM.
     */
    void get_kernel_sizes_brgemm(int32_t m,
                                 int32_t n,
                                 KernelSize &kernelsize_big,
                                 KernelSize &kernelsize_small,
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

        kernelsize_big.M = (m > 15) ? 16 : m;
        kernelsize_big.N = max_n_blocking;
        i_used_vector_reg_count_big = m_blocks * max_n_blocking;

        kernelsize_small.M = (m > 15) ? 16 : m;
        kernelsize_small.N = n % max_n_blocking;

        i_used_vector_reg_count_small = m_blocks * kernelsize_small.N;
    }

    /**
     * @brief Load a block of B with the given kernel size to vector registers
     */
    void generator_load_reg_block(Kernel &i_kernel,
                                  KernelSize &i_kernelsize,
                                  gpr_t i_register) {
        int32_t l_n_count = 0;
        int32_t l_reg_count = 0;

        // prepare C restore register Help 1
        i_kernel.add_instr(base_mov_imm(HELP_REG_1, i_kernelsize.N, 0));
        i_kernel.add_instr(base_mul_reg(HELP_REG_1, HELP_REG_1, LEADING_DIM_C_REG));

        for (; l_n_count < i_kernelsize.N; l_n_count++) {
            int32_t l_m_count = 0;
            while (l_m_count < i_kernelsize.M) {
                if ((i_kernelsize.M - l_m_count) > 15) {
                    i_kernel.add_instr(neon_ld1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                          i_register,
                                                          vector_count_t::vc4));
                    i_kernel.add_instr(base_add_imm(i_register,
                                                    i_register,
                                                    64,
                                                    0));
                    l_reg_count += 4;
                    l_m_count += 16;
                } else if ((i_kernelsize.M - l_m_count) > 11) {
                    i_kernel.add_instr(neon_ld1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                          i_register,
                                                          vector_count_t::vc3));
                    i_kernel.add_instr(base_add_imm(i_register,
                                                    i_register,
                                                    48,
                                                    0));
                    l_reg_count += 3;
                    l_m_count += 12;
                } else if ((i_kernelsize.M - l_m_count) > 7) {
                    i_kernel.add_instr(neon_ld1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                          i_register,
                                                          vector_count_t::vc2));
                    i_kernel.add_instr(base_add_imm(i_register,
                                                    i_register,
                                                    32,
                                                    0));
                    l_reg_count += 2;
                    l_m_count += 8;
                } else if ((i_kernelsize.M - l_m_count) > 3) {
                    i_kernel.add_instr(neon_ld1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                          i_register,
                                                          vector_count_t::vc1));
                    i_kernel.add_instr(base_add_imm(i_register,
                                                    i_register,
                                                    16,
                                                    0));
                    l_reg_count += 1;
                    l_m_count += 4;
                } else {
                    if ((i_kernelsize.M - l_m_count) > 1) {
                        i_kernel.add_instr(neon_ldr(static_cast<simd_fp_t>(l_reg_count),
                                                    i_register,
                                                    0,
                                                    arr_spec_t::d));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        8,
                                                        0));
                        l_m_count += 2;
                    }
                    if (((i_kernelsize.M - l_m_count) % 2) == 1) {
                        i_kernel.add_instr(neon_ld1_scalar_index(static_cast<simd_fp_t>(l_reg_count),
                                                                 i_register,
                                                                 (i_kernelsize.M % 4 == 3) ? 2 : 0));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        4,
                                                        0));
                        l_m_count++;
                    }
                    l_reg_count += 1;
                }
            }
            // Add Leading dimension
            i_kernel.add_instr(base_sub_imm(i_register,
                                            i_register,
                                            i_kernelsize.M * 4,
                                            0));
            i_kernel.add_instr(base_add_shifted_register(i_register,
                                                         i_register,
                                                         LEADING_DIM_C_REG,
                                                         0,
                                                         0));
        }
        // restore Working C reg for now with
        i_kernel.add_instr(base_sub_shifted_register(i_register,
                                                     i_register,
                                                     HELP_REG_1,
                                                     0,
                                                     0));
    }

    /**
     * @brief Store a block of B with the given kernel size to vector registers
     */
    void generator_store_reg_block(Kernel &i_kernel,
                                   KernelSize &i_kernelsize,
                                   gpr_t i_register) {
        int32_t l_n_count = 0;
        int32_t l_reg_count = 0;

        // prepare C restore register Help 1
        i_kernel.add_instr(base_mov_imm(HELP_REG_1, i_kernelsize.N, 0));
        i_kernel.add_instr(base_mul_reg(HELP_REG_1, HELP_REG_1, LEADING_DIM_C_REG));

        for (; l_n_count < i_kernelsize.N; l_n_count++) {
            int32_t l_m_count = 0;
            while (l_m_count < i_kernelsize.M) {
                if ((i_kernelsize.M - l_m_count) > 15) {
                    i_kernel.add_instr(neon_st1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                          i_register,
                                                          vector_count_t::vc4));
                    i_kernel.add_instr(base_add_imm(i_register,
                                                    i_register,
                                                    64,
                                                    0));
                    l_reg_count += 4;
                    l_m_count += 16;
                } else if ((i_kernelsize.M - l_m_count) > 11) {
                    i_kernel.add_instr(neon_st1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                          i_register,
                                                          vector_count_t::vc3));
                    i_kernel.add_instr(base_add_imm(i_register,
                                                    i_register,
                                                    48,
                                                    0));
                    l_reg_count += 3;
                    l_m_count += 12;
                } else if ((i_kernelsize.M - l_m_count) > 7) {
                    i_kernel.add_instr(neon_st1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                          i_register,
                                                          vector_count_t::vc2));
                    i_kernel.add_instr(base_add_imm(i_register,
                                                    i_register,
                                                    32,
                                                    0));
                    l_reg_count += 2;
                    l_m_count += 8;
                } else if ((i_kernelsize.M - l_m_count) > 3) {
                    i_kernel.add_instr(neon_st1_no_offset(static_cast<simd_fp_t>(l_reg_count),
                                                          i_register,
                                                          vector_count_t::vc1));
                    i_kernel.add_instr(base_add_imm(i_register,
                                                    i_register,
                                                    16,
                                                    0));
                    l_reg_count += 1;
                    l_m_count += 4;
                } else {
                    if ((i_kernelsize.M - l_m_count) > 1) {
                        i_kernel.add_instr(neon_str(static_cast<simd_fp_t>(l_reg_count),
                                                    i_register,
                                                    0,
                                                    arr_spec_t::d));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        8,
                                                        0));
                        l_m_count += 2;
                    }
                    if (((i_kernelsize.M - l_m_count) % 2) == 1) {
                        i_kernel.add_instr(neon_st1_scalar_index(static_cast<simd_fp_t>(l_reg_count),
                                                                 i_register,
                                                                 (i_kernelsize.M % 4 == 3) ? 2 : 0));
                        i_kernel.add_instr(base_add_imm(i_register,
                                                        i_register,
                                                        4,
                                                        0));
                        l_m_count++;
                    }

                    l_reg_count += 1;
                }
            }
            // Add Leading dimension
            i_kernel.add_instr(base_sub_imm(i_register,
                                            i_register,
                                            i_kernelsize.M * 4,
                                            0));
            i_kernel.add_instr(base_add_shifted_register(i_register,
                                                         i_register,
                                                         LEADING_DIM_C_REG,
                                                         0,
                                                         0));
        }
        // restore Working C reg for now with
        i_kernel.add_instr(base_sub_shifted_register(WORKING_ADDRESS_C_REG,
                                                     WORKING_ADDRESS_C_REG,
                                                     HELP_REG_1,
                                                     0,
                                                     0));
    }

}  // namespace TenGen::MiniJit::Generator::Util

#endif  // TENGEN_MINI_JIT_GENERATOR_UTIL_H
