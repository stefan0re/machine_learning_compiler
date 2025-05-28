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

    void Unary::gen_unary_transpose(uint32_t m, uint32_t n) {
        int max_size = m * n;
        int helper = 0;

        // for all elements in a
        for (int i = 0; i < max_size; i++) {
            // copy from a to b
            // TODO: This instructions are not implemented
            // ldr x11, [x7]
            m_kernel.add_instr(3107979499);
            // str x11, [x8]
            m_kernel.add_instr(3103785227);

            // calc new element_b by adding the offset (size)
            m_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_B_REG, Util::WORKING_ADDRESS_B_REG, m * 4, 0));
            helper += m;

            // if the elements_b exeeds the maximum size
            if (max_size - helper < 0) {
                // start over
                m_kernel.add_instr(inst::InstGen::base_sub_imm(Util::WORKING_ADDRESS_B_REG, Util::WORKING_ADDRESS_B_REG, max_size * 4, 0));
                helper -= max_size;

                // next element in b
                m_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_B_REG, Util::WORKING_ADDRESS_B_REG, 4, 0));
                helper += 1;
            }

            // next element in a
            m_kernel.add_instr(inst::InstGen::base_add_imm(Util::WORKING_ADDRESS_A_REG, Util::WORKING_ADDRESS_A_REG, 4, 0));
        }
    }

    Unary::error_t Unary::generate(uint32_t m,
                                   uint32_t n,
                                   uint32_t trans_b,
                                   Unary::dtype_t dtype,
                                   Unary::ptype_t ptype) {
        // safely calculate number of iterations for main loop and number of rest elements
        uint64_t total = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
        uint32_t iterations = static_cast<uint32_t>((total - (total % 4)) / 16);
        uint32_t rest = static_cast<uint32_t>(total % 16);

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
        m_kernel.add_instr(inst::InstGen::base_lsl_imm(inst::InstGen::x2, inst::InstGen::x2, 2));
        m_kernel.add_instr(inst::InstGen::base_lsl_imm(inst::InstGen::x3, inst::InstGen::x3, 2));

        if (!(ptype == Unary::ptype_t::identity) && !(trans_b == 1)) {
            // move 0 to v31 for relu
            m_kernel.add_instr(inst::InstGen::neon_movi_zero(inst::InstGen::simd_fp_t::v31, true, false));

            // generate main loop
            if (iterations > 0) {
                // set loop counter, if number of iterations too high for immediate use movk
                uint64_t value = iterations;
                uint16_t lo = value & 0xffff;
                uint16_t hi = (value >> 16) & 0xffff;

                m_kernel.add_instr(inst::InstGen::base_movz(inst::InstGen::x9, lo, 0));  // movz x9, lo
                if (hi != 0)
                    m_kernel.add_instr(inst::InstGen::base_movk(inst::InstGen::x9, hi, 16));  // movk x9, hi, LSL #16

                // loop
                size_t loop_count = m_kernel.get_size();

                m_kernel.add_instr(
                    inst::InstGen::base_sub_imm(
                        inst::InstGen::x9,
                        inst::InstGen::x9,
                        1,
                        0));

                m_kernel.add_instr(inst::InstGen::neon_ld1_multiple(inst::InstGen::v0,
                                                                    inst::InstGen::x7,
                                                                    inst::InstGen::ld1_opcode_t::four_regs,
                                                                    inst::InstGen::ld1_t::S4));

                if (ptype == Unary::ptype_t::zero) {
                    m_kernel.add_instr(inst::InstGen::neon_movi_zero(inst::InstGen::v0, true, false));
                    m_kernel.add_instr(inst::InstGen::neon_movi_zero(inst::InstGen::v1, true, false));
                    m_kernel.add_instr(inst::InstGen::neon_movi_zero(inst::InstGen::v2, true, false));
                    m_kernel.add_instr(inst::InstGen::neon_movi_zero(inst::InstGen::v3, true, false));
                    this->fops += 4;
                } else if (ptype == Unary::ptype_t::relu) {
                    m_kernel.add_instr(inst::InstGen::neon_fmax_vector(inst::InstGen::v0,
                                                                       inst::InstGen::v0,
                                                                       inst::InstGen::simd_fp_t::v31,
                                                                       false));
                    m_kernel.add_instr(inst::InstGen::neon_fmax_vector(inst::InstGen::v1,
                                                                       inst::InstGen::v1,
                                                                       inst::InstGen::simd_fp_t::v31,
                                                                       false));
                    m_kernel.add_instr(inst::InstGen::neon_fmax_vector(inst::InstGen::v2,
                                                                       inst::InstGen::v2,
                                                                       inst::InstGen::simd_fp_t::v31,
                                                                       false));
                    m_kernel.add_instr(inst::InstGen::neon_fmax_vector(inst::InstGen::v3,
                                                                       inst::InstGen::v3,
                                                                       inst::InstGen::simd_fp_t::v31,
                                                                       false));
                    this->fops += 4;
                }

                m_kernel.add_instr(inst::InstGen::neon_st1_multiple(inst::InstGen::v0,
                                                                    inst::InstGen::x8,
                                                                    inst::InstGen::ld1_opcode_t::four_regs,
                                                                    inst::InstGen::ld1_t::S4));

                m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7, inst::InstGen::x7, 4 * 16, 0));
                m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x8, inst::InstGen::x8, 4 * 16, 0));

                // jump loop
                m_kernel.add_instr(inst::InstGen::base_br_cbnz(inst::InstGen::x9, (loop_count - m_kernel.get_size()) / 4));
            }

            // try to use ld1 with as many registers as possible for rest (rest in [0, 15])
            uint32_t next_bigger = rest;

            while (next_bigger % 4 != 0) {
                next_bigger--;
            }

            inst::InstGen::ld1_opcode_t num_regs;

            if (next_bigger == 12) {
                num_regs = inst::InstGen::ld1_opcode_t::three_regs;
            } else if (next_bigger == 8) {
                num_regs = inst::InstGen::ld1_opcode_t::two_regs;
            } else if (next_bigger == 4) {
                num_regs = inst::InstGen::ld1_opcode_t::one_regs;
            }

            if (next_bigger > 0) {
                m_kernel.add_instr(inst::InstGen::neon_ld1_multiple(inst::InstGen::v0,
                                                                    inst::InstGen::x7,
                                                                    num_regs,
                                                                    inst::InstGen::ld1_t::S4));

                int32_t reg_count = 0;

                for (int i = 0; i < (int)(next_bigger / 4); i++) {
                    if (ptype == Unary::ptype_t::zero) {
                        m_kernel.add_instr(inst::InstGen::neon_movi_zero(static_cast<inst::InstGen::simd_fp_t>(reg_count++), true, false));
                    } else if (ptype == Unary::ptype_t::relu) {
                        m_kernel.add_instr(inst::InstGen::neon_fmax_vector(static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                           static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                           inst::InstGen::simd_fp_t::v31,
                                                                           false));
                        reg_count++;
                    }
                }

                m_kernel.add_instr(inst::InstGen::neon_st1_multiple(inst::InstGen::v0,
                                                                    inst::InstGen::x8,
                                                                    num_regs,
                                                                    inst::InstGen::ld1_t::S4));

                m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7, inst::InstGen::x7, 4 * next_bigger, 0));
                m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x8, inst::InstGen::x8, 4 * next_bigger, 0));
            }

            // final rest (in [0, 3]) with single ld1 statements per element
            int32_t reg_count = 0;
            rest = (uint32_t)std::abs((int)next_bigger - (int)rest);
            for (int i = 0; i < rest; i++) {
                m_kernel.add_instr(
                    inst::InstGen::neon_ld1_no_offset(
                        static_cast<inst::InstGen::simd_fp_t>(i),
                        inst::InstGen::x7,
                        inst::InstGen::vector_count_t::vc1));

                if (ptype == Unary::ptype_t::zero) {
                    m_kernel.add_instr(inst::InstGen::neon_movi_zero(static_cast<inst::InstGen::simd_fp_t>(reg_count++), true, false));
                } else if (ptype == Unary::ptype_t::relu) {
                    m_kernel.add_instr(inst::InstGen::neon_fmax_vector(static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                       static_cast<inst::InstGen::simd_fp_t>(reg_count),
                                                                       inst::InstGen::simd_fp_t::v31,
                                                                       false));
                    reg_count++;
                }
                m_kernel.add_instr(
                    inst::InstGen::neon_st1_no_offset(
                        static_cast<inst::InstGen::simd_fp_t>(i),
                        inst::InstGen::x8,
                        inst::InstGen::vector_count_t::vc1));

                // advance the base pointer by 1 elements
                m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x7, inst::InstGen::x7, 4, 0));
                m_kernel.add_instr(inst::InstGen::base_add_imm(inst::InstGen::x8, inst::InstGen::x8, 4, 0));
            }

            // Transpose
        } else {
            gen_unary_transpose(m, n);
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
        return reinterpret_cast<kernel_t>(m_kernel.get_kernel());
    }
}  // namespace mini_jit::generator