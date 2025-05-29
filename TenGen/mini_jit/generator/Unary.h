#ifndef TENGEN_MINI_JIT_GENERATOR_UNARY_H
#define TENGEN_MINI_JIT_GENERATOR_UNARY_H

#include <cstdint>

#include "TenGen/mini_jit/backend/Kernel.h"
#include "TenGen/mini_jit/instructions/Encoding.h"
#include "TenGen/types/Structs.h"
#include "TenGen/types/Types.h"

using namespace TenGen::Types;
using namespace TenGen::Structs;
using namespace TenGen::MiniJit::Instructions::Encoding;
using Kernel = TenGen::MiniJit::Backend::Kernel;

namespace TenGen::MiniJit::Generator {

    class Unary {
       private:
        Kernel m_kernel;

        void gen_unary_transpose(uint32_t m, uint32_t n) {
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
                m_kernel.add_instr(base_add_imm(WORKING_ADDRESS_B_REG, WORKING_ADDRESS_B_REG, m * 4, 0));
                helper += m;

                // if the elements_b exeeds the maximum size
                if (max_size - helper < 0) {
                    // start over
                    m_kernel.add_instr(base_sub_imm(WORKING_ADDRESS_B_REG, WORKING_ADDRESS_B_REG, max_size * 4, 0));
                    helper -= max_size;

                    // next element in b
                    m_kernel.add_instr(base_add_imm(WORKING_ADDRESS_B_REG, WORKING_ADDRESS_B_REG, 4, 0));
                    helper += 1;
                }

                // next element in a
                m_kernel.add_instr(base_add_imm(WORKING_ADDRESS_A_REG, WORKING_ADDRESS_A_REG, 4, 0));
            }
        }

        void gen_unary_relu() {
            // TODO: Implement relu generation
        }

        void gen_unary_zero() {
            // TODO: Implement zero generation
        }

       public:
        int32_t fops = 0;

        /**
         * @brief Generate a kernel for a unary primitive.
         * @param m       Number of rows in A and B.
         * @param n       Number of columns in A and B.
         * @param trans_b 0 if B is stored in column-major order, 1 if B is stored in row-major order.
         * @param dtype   Data type of the matrices.
         * @param ptype   Primitive type.
         * @return error_t::success on success, another error_t value otherwise.
         **/
        TenGen::Types::error_t generate(uint32_t m,
                                        uint32_t n,
                                        uint32_t trans_b,
                                        dtype_t dtype,
                                        ptype_t ptype) {
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
            m_kernel.add_instr(base_mov_register(x7,
                                                 x0));
            m_kernel.add_instr(base_mov_register(x8,
                                                 x1));

            // shift leading dimensions to 4 bytes
            m_kernel.add_instr(base_lsl_imm(x2, x2, 2));
            m_kernel.add_instr(base_lsl_imm(x3, x3, 2));

            if (!(ptype == ptype_t::identity) && !(trans_b == 1)) {
                // move 0 to v31 for relu
                m_kernel.add_instr(neon_movi_zero(simd_fp_t::v31, true, false));

                // generate main loop
                if (iterations > 0) {
                    // set loop counter, if number of iterations too high for immediate use movk
                    uint64_t value = iterations;
                    uint16_t lo = value & 0xffff;
                    uint16_t hi = (value >> 16) & 0xffff;

                    m_kernel.add_instr(base_movz(x9, lo, 0));  // movz x9, lo
                    if (hi != 0)
                        m_kernel.add_instr(base_movk(x9, hi, 16));  // movk x9, hi, LSL #16

                    // loop
                    size_t loop_count = m_kernel.get_size();

                    m_kernel.add_instr(
                        base_sub_imm(
                            x9,
                            x9,
                            1,
                            0));

                    m_kernel.add_instr(neon_ld1_multiple(v0,
                                                         x7,
                                                         ld1_opcode_t::four_regs,
                                                         ld1_t::S4));

                    if (ptype == ptype_t::zero) {
                        m_kernel.add_instr(neon_movi_zero(v0, true, false));
                        m_kernel.add_instr(neon_movi_zero(v1, true, false));
                        m_kernel.add_instr(neon_movi_zero(v2, true, false));
                        m_kernel.add_instr(neon_movi_zero(v3, true, false));
                        this->fops += 4;
                    } else if (ptype == ptype_t::relu) {
                        m_kernel.add_instr(neon_fmax_vector(v0,
                                                            v0,
                                                            simd_fp_t::v31,
                                                            false));
                        m_kernel.add_instr(neon_fmax_vector(v1,
                                                            v1,
                                                            simd_fp_t::v31,
                                                            false));
                        m_kernel.add_instr(neon_fmax_vector(v2,
                                                            v2,
                                                            simd_fp_t::v31,
                                                            false));
                        m_kernel.add_instr(neon_fmax_vector(v3,
                                                            v3,
                                                            simd_fp_t::v31,
                                                            false));
                        this->fops += 4;
                    }

                    m_kernel.add_instr(neon_st1_multiple(v0,
                                                         x8,
                                                         ld1_opcode_t::four_regs,
                                                         ld1_t::S4));

                    m_kernel.add_instr(base_add_imm(x7, x7, 4 * 16, 0));
                    m_kernel.add_instr(base_add_imm(x8, x8, 4 * 16, 0));

                    // jump loop
                    m_kernel.add_instr(base_br_cbnz(x9, (loop_count - m_kernel.get_size()) / 4));
                }

                // try to use ld1 with as many registers as possible for rest (rest in [0, 15])
                uint32_t next_bigger = rest;

                while (next_bigger % 4 != 0) {
                    next_bigger--;
                }

                ld1_opcode_t num_regs;

                if (next_bigger == 12) {
                    num_regs = ld1_opcode_t::three_regs;
                } else if (next_bigger == 8) {
                    num_regs = ld1_opcode_t::two_regs;
                } else if (next_bigger == 4) {
                    num_regs = ld1_opcode_t::one_regs;
                }

                if (next_bigger > 0) {
                    m_kernel.add_instr(neon_ld1_multiple(v0,
                                                         x7,
                                                         num_regs,
                                                         ld1_t::S4));

                    int32_t reg_count = 0;

                    for (int i = 0; i < (int)(next_bigger / 4); i++) {
                        if (ptype == ptype_t::zero) {
                            m_kernel.add_instr(neon_movi_zero(static_cast<simd_fp_t>(reg_count++), true, false));
                        } else if (ptype == ptype_t::relu) {
                            m_kernel.add_instr(neon_fmax_vector(static_cast<simd_fp_t>(reg_count),
                                                                static_cast<simd_fp_t>(reg_count),
                                                                simd_fp_t::v31,
                                                                false));
                            reg_count++;
                        }
                    }

                    m_kernel.add_instr(neon_st1_multiple(v0,
                                                         x8,
                                                         num_regs,
                                                         ld1_t::S4));

                    m_kernel.add_instr(base_add_imm(x7, x7, 4 * next_bigger, 0));
                    m_kernel.add_instr(base_add_imm(x8, x8, 4 * next_bigger, 0));
                }

                // final rest (in [0, 3]) with single ld1 statements per element
                int32_t reg_count = 0;
                rest = (uint32_t)std::abs((int)next_bigger - (int)rest);
                for (int i = 0; i < rest; i++) {
                    m_kernel.add_instr(
                        neon_ld1_no_offset(
                            static_cast<simd_fp_t>(i),
                            x7,
                            vector_count_t::vc1));

                    if (ptype == ptype_t::zero) {
                        m_kernel.add_instr(neon_movi_zero(static_cast<simd_fp_t>(reg_count++), true, false));
                    } else if (ptype == ptype_t::relu) {
                        m_kernel.add_instr(neon_fmax_vector(static_cast<simd_fp_t>(reg_count),
                                                            static_cast<simd_fp_t>(reg_count),
                                                            simd_fp_t::v31,
                                                            false));
                        reg_count++;
                    }
                    m_kernel.add_instr(
                        neon_st1_no_offset(
                            static_cast<simd_fp_t>(i),
                            x8,
                            vector_count_t::vc1));

                    // advance the base pointer by 1 elements
                    m_kernel.add_instr(base_add_imm(x7, x7, 4, 0));
                    m_kernel.add_instr(base_add_imm(x8, x8, 4, 0));
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
            m_kernel.add_instr(base_ret());

            m_kernel.set_kernel();

            m_kernel.write("output_test.bin");

            return TenGen::Types::error_t::success;
        }

        /*
         * Kernel type.
         * The kernel is a function that takes the following parameters:
         * - a:    Pointer to column-major matrix A, nullptr if zero kernel.
         * - b:    Pointer to matrix B.
         * - ld_a: Leading dimension of A.
         * - ld_b: Leading dimension of B.
         */
        using kernel_t = void (*)(void const* a,
                                  void* b,
                                  int64_t ld_a,
                                  int64_t ld_b);

        /**
         * @brief Get the generated kernel: B := op(A).
         * @return pointer to the generated kernel.
         **/
        kernel_t get_kernel() const {
            return reinterpret_cast<kernel_t>(m_kernel.get_kernel());
        }
    };

}  // namespace TenGen::MiniJit::Generator
#endif  // TENGEN_MINI_JIT_GENERATOR_UNARY_H