#include "Unary.h"

#include <float.h>
#include <math.h>

#include <iostream>

#include "../instructions/instructions.h"
#include "Util.h"

namespace inst = mini_jit::instructions;

namespace mini_jit::generator {
    typedef struct {
        int32_t m;
        int32_t n;
        int32_t m_iters;
        int32_t n_iters;
        uint32_t offset;
        Util::KernelSize kernelsize;
    } AreaDefinition;

    mini_jit::backend::Kernel Unary::m_kernel;

    void Unary::gen_transpose_micro_4x4(uint32_t i_m,
                                        uint32_t i_n) {
        // ldr
        for (size_t i = 0; i < 4; i++) {
            m_kernel.add_instr(inst::InstGen::neon_ld1_no_offset(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i),
                                                                 inst::InstGen::x0,
                                                                 inst::InstGen::vector_count_t::vc1));
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x0,
                                                                        inst::InstGen::x0,
                                                                        inst::InstGen::x2,
                                                                        0,
                                                                        0));
        }
        /* first part */
        // trn
        m_kernel.add_instr(inst::InstGen::neon_trn(inst::InstGen::v4, inst::InstGen::v0, inst::InstGen::v1, 1));
        m_kernel.add_instr(inst::InstGen::neon_trn(inst::InstGen::v5, inst::InstGen::v0, inst::InstGen::v1, 2));
        m_kernel.add_instr(inst::InstGen::neon_trn(inst::InstGen::v6, inst::InstGen::v2, inst::InstGen::v3, 1));
        m_kernel.add_instr(inst::InstGen::neon_trn(inst::InstGen::v7, inst::InstGen::v2, inst::InstGen::v3, 2));

        // zip
        m_kernel.add_instr(inst::InstGen::neon_zip(inst::InstGen::v8, inst::InstGen::v4, inst::InstGen::v6, 1));
        m_kernel.add_instr(inst::InstGen::neon_zip(inst::InstGen::v9, inst::InstGen::v5, inst::InstGen::v7, 1));
        m_kernel.add_instr(inst::InstGen::neon_zip(inst::InstGen::v10, inst::InstGen::v4, inst::InstGen::v6, 2));
        m_kernel.add_instr(inst::InstGen::neon_zip(inst::InstGen::v11, inst::InstGen::v5, inst::InstGen::v7, 2));

        // str
        for (size_t i = 0; i < 4; i++) {
            m_kernel.add_instr(inst::InstGen::neon_st1_no_offset(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v8 + i),
                                                                 inst::InstGen::x1,
                                                                 inst::InstGen::vector_count_t::vc1));
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x1,
                                                                        inst::InstGen::x1,
                                                                        inst::InstGen::x3,
                                                                        0,
                                                                        0));
        }
    }

    void Unary::gen_transpose_micro_reminder(uint32_t i_m,
                                             uint32_t i_n) {
        // load each value to seperate register
        int32_t v_reg_count = 0;
        for (size_t l_n = 0; l_n < i_n; l_n++) {
            for (size_t l_m = 0; l_m < i_m; l_m++) {
                m_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(v_reg_count++),
                                                           inst::InstGen::x0,
                                                           4,
                                                           inst::InstGen::arr_spec_t::s));
            }
            // set to next column
            m_kernel.add_instr(inst::InstGen::base_sub_imm(inst::InstGen::x0, inst::InstGen::x0, 4 * i_m, 0));
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x0,
                                                                        inst::InstGen::x0,
                                                                        inst::InstGen::x2,
                                                                        0,
                                                                        0));
        }
        v_reg_count = 0;
        // store values from seperate register
        for (size_t l_m = 0; l_m < i_m; l_m++) {
            for (size_t l_n = 0; l_n < i_n; l_n++) {
                m_kernel.add_instr(inst::InstGen::neon_str(static_cast<inst::InstGen::simd_fp_t>(v_reg_count),
                                                           inst::InstGen::x1,
                                                           4,
                                                           inst::InstGen::arr_spec_t::s));
                v_reg_count += i_m;
            }
            m_kernel.add_instr(inst::InstGen::base_sub_imm(inst::InstGen::x1, inst::InstGen::x1, 4 * i_n, 0));
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x1,
                                                                        inst::InstGen::x1,
                                                                        inst::InstGen::x3,
                                                                        0,
                                                                        0));
            v_reg_count -= i_m * i_n;
            v_reg_count += 1;
        }
    }

    void Unary::gen_transpose(uint32_t i_m,
                              uint32_t i_n) {
        /* get blocking */
        uint32_t m_blocks_full = i_m / 4;
        uint32_t m_blocks_reminder = i_m % 4;

        uint32_t n_blocks_full = i_n / 4;
        uint32_t n_blocks_reminder = i_n % 4;

        // write M and N to x12
        m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x12, m_blocks_full, 0));

        // write 16 to x13
        m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x13, 16, 0));

        // mul x13 to x12
        m_kernel.add_instr(inst::InstGen::base_mul_reg(inst::InstGen::x12,
                                                       inst::InstGen::x12,
                                                       inst::InstGen::x13));

        // write restore size to x5 for A and x6 for B
        m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x5, 4, 0));
        m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x6, i_m, 0));
        m_kernel.add_instr(inst::InstGen::base_mul_reg(inst::InstGen::x5,
                                                       inst::InstGen::x5,
                                                       inst::InstGen::x2));
        m_kernel.add_instr(inst::InstGen::base_mul_reg(inst::InstGen::x6,
                                                       inst::InstGen::x6,
                                                       inst::InstGen::x3));
        // set M 10 and N 11 loop
        m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x11, n_blocks_full, 0));

        m_kernel.add_instr(inst::InstGen::base_sub_imm(inst::InstGen::x11, inst::InstGen::x11, 1, 0));

        std::size_t n_loop_pos = m_kernel.get_size();
        m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x10, m_blocks_full, 0));

        m_kernel.add_instr(inst::InstGen::base_sub_imm(inst::InstGen::x10, inst::InstGen::x10, 1, 0));

        std::size_t m_loop_pos = m_kernel.get_size();
        gen_transpose_micro_4x4(4, 4);

        m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(inst::InstGen::x0,
                                                                    inst::InstGen::x0,
                                                                    inst::InstGen::x5,
                                                                    0,
                                                                    0));

        m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x0,
                                                       inst::InstGen::x0,
                                                       4 * 4,
                                                       0));
        m_kernel.add_instr(inst::InstGen::base_br_cbnz(inst::InstGen::x10,
                                                       (m_loop_pos - m_kernel.get_size()) / 4 - 1));
        if (m_blocks_reminder > 0) {
            gen_transpose_micro_reminder(m_blocks_reminder, 4);
        } else {
            // if no reminder, we need to adjust a pointer
            m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x0,
                                                                        inst::InstGen::x0,
                                                                        inst::InstGen::x5,
                                                                        0,
                                                                        0));
        }

        // adjust a and b pointer
        m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(inst::InstGen::x0,
                                                                    inst::InstGen::x0,
                                                                    inst::InstGen::x12,
                                                                    0,
                                                                    0));

        m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(inst::InstGen::x1,
                                                                    inst::InstGen::x1,
                                                                    inst::InstGen::x6,
                                                                    0,
                                                                    0));

        m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x1,
                                                       inst::InstGen::x1,
                                                       4 * 4,
                                                       0));
        m_kernel.add_instr(inst::InstGen::base_br_cbnz(inst::InstGen::x11,
                                                       (n_loop_pos - m_kernel.get_size()) / 4 - 1));

        // handle reminder N column
        if (n_blocks_reminder > 0) {
            m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x5, n_blocks_reminder, 0));
            m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x6, i_m, 0));
            m_kernel.add_instr(inst::InstGen::base_mul_reg(inst::InstGen::x5,
                                                           inst::InstGen::x5,
                                                           inst::InstGen::x2));
            for (uint32_t l_m = 0; l_m < m_blocks_full; l_m++) {
                gen_transpose_micro_reminder(4, n_blocks_reminder);

                m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(inst::InstGen::x0,
                                                                            inst::InstGen::x0,
                                                                            inst::InstGen::x5,
                                                                            0,
                                                                            0));

                m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x0,
                                                               inst::InstGen::x0,
                                                               4 * 4,
                                                               0));
            }
            gen_transpose_micro_reminder(m_blocks_reminder, n_blocks_reminder);
        }
    }

    void Unary::gen_zero(uint32_t m,
                         uint32_t n) {
        for (uint32_t i = 0; i < 4; i++) {
            m_kernel.add_instr(inst::InstGen::neon_eor(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i),
                                                       static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i),
                                                       static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i)));
        }
        // help for N loops
        m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x13, m * 4, 0));

        m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(inst::InstGen::x13,
                                                                    inst::InstGen::x3,
                                                                    inst::InstGen::x13,
                                                                    0,
                                                                    0));

        // set N loop counter
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::N_LOOP_COUNT_REG, n, 0));
        // sub N loop register
        m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::N_LOOP_COUNT_REG,
                                                       Util::N_LOOP_COUNT_REG,
                                                       1,
                                                       0));

        std::size_t loop_pos_n = m_kernel.get_size();

        if ((m / 16) > 0) {
            // set M loop counter
            m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::M_LOOP_COUNT_REG, m / 16, 0));
            // sub M loop register
            m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::M_LOOP_COUNT_REG,
                                                           Util::M_LOOP_COUNT_REG,
                                                           1,
                                                           0));

            // inner store loop
            std::size_t loop_pos_m = m_kernel.get_size();
            for (size_t i = 0; i < 4; i++) {
                m_kernel.add_instr(inst::InstGen::neon_str(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + (i % 4)), inst::InstGen::x1, 16, inst::InstGen::arr_spec_t::q));
            }

            // BRANCH M
            m_kernel.add_instr(inst::InstGen::base_br_cbnz(Util::M_LOOP_COUNT_REG,
                                                           (loop_pos_m - m_kernel.get_size()) / 4 - 1));
        }
        for (size_t i = 0; i < (m % 16); i++) {
            m_kernel.add_instr(inst::InstGen::neon_str(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + (i % 4)), inst::InstGen::x1, 4, inst::InstGen::arr_spec_t::s));
        }

        // BRANCH N
        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x1, inst::InstGen::x1, inst::InstGen::x13, 0, 0));
        m_kernel.add_instr(inst::InstGen::base_br_cbnz(Util::N_LOOP_COUNT_REG,
                                                       (loop_pos_n - m_kernel.get_size()) / 4 - 1));
    }

    // assuming that vr 31 is not used by C accumulator
    void Unary::gen_relu(uint32_t m,
                         uint32_t n) {
        m_kernel.add_instr(inst::InstGen::neon_eor(inst::InstGen::v31,
                                                   inst::InstGen::v31,
                                                   inst::InstGen::v31));
        // help for N loops
        m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x13, m * 4, 0));

        m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(inst::InstGen::x13,
                                                                    inst::InstGen::x3,
                                                                    inst::InstGen::x13,
                                                                    0,
                                                                    0));
        m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(inst::InstGen::x14,
                                                                    inst::InstGen::x2,
                                                                    inst::InstGen::x13,
                                                                    0,
                                                                    0));

        int reg_for_m_col = m / 4;
        int use_loop = reg_for_m_col / 31;
        reg_for_m_col -= use_loop * 31;

        // set N loop counter
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::N_LOOP_COUNT_REG, n, 0));
        // sub N loop register
        m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::N_LOOP_COUNT_REG,
                                                       Util::N_LOOP_COUNT_REG,
                                                       1,
                                                       0));
        std::size_t loop_pos_n = m_kernel.get_size();

        for (; use_loop > -1; use_loop--) {
            if (use_loop > 0) {
                for (size_t i = 0; i < 31; i++) {
                    m_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), inst::InstGen::x7, 16, inst::InstGen::arr_spec_t::q));
                }
            } else {
                for (size_t i = 0; i < reg_for_m_col; i++) {
                    m_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), inst::InstGen::x7, 16, inst::InstGen::arr_spec_t::q));
                }
                for (size_t i = 0; i < (m % 4); i++) {
                    m_kernel.add_instr(inst::InstGen::neon_ld1_scalar_index(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + reg_for_m_col), inst::InstGen::x7, i));
                    m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7, inst::InstGen::x7, 4, 0));
                }
            }

            // USE relu
            if (use_loop > 0) {
                for (size_t i = 0; i < 31; i++) {
                    m_kernel.add_instr(inst::InstGen::neon_fmax_vector(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v31), false));
                }
            } else {
                for (size_t i = 0; i < reg_for_m_col; i++) {
                    m_kernel.add_instr(inst::InstGen::neon_fmax_vector(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v31), false));
                }
            }

            if (use_loop > 0) {
                for (size_t i = 0; i < 31; i++) {
                    m_kernel.add_instr(inst::InstGen::neon_str(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), inst::InstGen::x8, 16, inst::InstGen::arr_spec_t::q));
                }
            } else {
                for (size_t i = 0; i < reg_for_m_col; i++) {
                    m_kernel.add_instr(inst::InstGen::neon_str(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), inst::InstGen::x8, 16, inst::InstGen::arr_spec_t::q));
                }
                for (size_t i = 0; i < (m % 4); i++) {
                    m_kernel.add_instr(inst::InstGen::neon_st1_scalar_index(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + reg_for_m_col), inst::InstGen::x8, i));
                    m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x8, inst::InstGen::x8, 4, 0));
                }
            }
        }
        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x0, inst::InstGen::x0, inst::InstGen::x14, 0, 0));
        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x1, inst::InstGen::x1, inst::InstGen::x13, 0, 0));
        m_kernel.add_instr(inst::InstGen::base_br_cbnz(Util::N_LOOP_COUNT_REG,
                                                       (loop_pos_n - m_kernel.get_size()) / 4 - 1));
    }

    void Unary::gen_identity(uint32_t m,
                             uint32_t n) {
        // help for N loops
        m_kernel.add_instr(inst::InstGen::base_mov_imm(inst::InstGen::x13, m * 4, 0));

        m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(inst::InstGen::x13,
                                                                    inst::InstGen::x3,
                                                                    inst::InstGen::x13,
                                                                    0,
                                                                    0));
        m_kernel.add_instr(inst::InstGen::base_sub_shifted_register(inst::InstGen::x14,
                                                                    inst::InstGen::x2,
                                                                    inst::InstGen::x13,
                                                                    0,
                                                                    0));

        int reg_for_m_col = m / 4;
        int use_loop = reg_for_m_col / 32;
        reg_for_m_col -= use_loop * 32;

        // set N loop counter
        m_kernel.add_instr(inst::InstGen::base_mov_imm(Util::N_LOOP_COUNT_REG, n, 0));
        // sub N loop register
        m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::N_LOOP_COUNT_REG,
                                                       Util::N_LOOP_COUNT_REG,
                                                       1,
                                                       0));
        std::size_t loop_pos_n = m_kernel.get_size();

        for (; use_loop > -1; use_loop--) {
            if (use_loop > 0) {
                for (size_t i = 0; i < 32; i++) {
                    m_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), inst::InstGen::x7, 16, inst::InstGen::arr_spec_t::q));
                }
            } else {
                for (size_t i = 0; i < reg_for_m_col; i++) {
                    m_kernel.add_instr(inst::InstGen::neon_ldr(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), inst::InstGen::x7, 16, inst::InstGen::arr_spec_t::q));
                }
                for (size_t i = 0; i < (m % 4); i++) {
                    m_kernel.add_instr(inst::InstGen::neon_ld1_scalar_index(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + reg_for_m_col), inst::InstGen::x7, i));
                    m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7, inst::InstGen::x7, 4, 0));
                }
            }

            if (use_loop > 0) {
                for (size_t i = 0; i < 32; i++) {
                    m_kernel.add_instr(inst::InstGen::neon_str(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), inst::InstGen::x8, 16, inst::InstGen::arr_spec_t::q));
                }
            } else {
                for (size_t i = 0; i < reg_for_m_col; i++) {
                    m_kernel.add_instr(inst::InstGen::neon_str(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + i), inst::InstGen::x8, 16, inst::InstGen::arr_spec_t::q));
                }
                for (size_t i = 0; i < (m % 4); i++) {
                    m_kernel.add_instr(inst::InstGen::neon_st1_scalar_index(static_cast<inst::InstGen::simd_fp_t>(inst::InstGen::v0 + reg_for_m_col), inst::InstGen::x8, i));
                    m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x8, inst::InstGen::x8, 4, 0));
                }
            }
        }
        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x0, inst::InstGen::x0, inst::InstGen::x14, 0, 0));
        m_kernel.add_instr(inst::InstGen::base_add_shifted_register(inst::InstGen::x1, inst::InstGen::x1, inst::InstGen::x13, 0, 0));
        m_kernel.add_instr(inst::InstGen::base_br_cbnz(Util::N_LOOP_COUNT_REG,
                                                       (loop_pos_n - m_kernel.get_size()) / 4 - 1));
    }

    Unary::error_t Unary::generate(uint32_t m,
                                   uint32_t n,
                                   Unary::dtype_t dtype,
                                   Unary::ptype_t ptype) {
        // procedure call standard (store to stack)
        m_kernel.add_instr(0x6DBF27E8);
        m_kernel.add_instr(0x6DBF2FEA);
        m_kernel.add_instr(0x6DBF37EC);
        m_kernel.add_instr(0x6DBF3FEE);

        //  Store pointers of A and B to x7, x8
        m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x7,
                                                            inst::InstGen::x0));
        m_kernel.add_instr(inst::InstGen::base_mov_register(inst::InstGen::x8,
                                                            inst::InstGen::x1));

        // shift leading dimensions to 4 bytes
        m_kernel.add_instr(0xd37ef442);
        m_kernel.add_instr(0xd37ef463);

        if (ptype == Unary::ptype_t::trans) {
            gen_transpose(m,
                          n);
        } else if (ptype == Unary::ptype_t::zero) {
            gen_zero(m, n);
        } else if (ptype == Unary::ptype_t::identity) {
            gen_identity(m, n);
        } else if (ptype == Unary::ptype_t::relu) {
            gen_relu(m, n);
        }

        // procedure call standard (load from stack)
        m_kernel.add_instr(0x6CC13FEE);
        m_kernel.add_instr(0x6CC137EC);
        m_kernel.add_instr(0x6CC12FEA);
        m_kernel.add_instr(0x6CC127E8);

        // ret
        m_kernel.add_instr(mini_jit::instructions::InstGen::base_ret());

        m_kernel.set_kernel();

        m_kernel.write("output_test.bin");

        return Unary::error_t::success;
    }

    mini_jit::generator::Unary::kernel_t mini_jit::generator::Unary::get_kernel() const {
        return reinterpret_cast<kernel_t>(const_cast<void*>(m_kernel.get_kernel()));
    }
}  // namespace mini_jit::generator