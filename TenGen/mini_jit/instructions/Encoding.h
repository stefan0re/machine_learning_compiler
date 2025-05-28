#ifndef TENGEN_MINI_JIT_INSTRUCTIONS_ENCODING_H
#define TENGEN_MINI_JIT_INSTRUCTIONS_ENCODING_H

#include <math.h>

#include <cstdint>
#include <string>

#include "TenGen.h"

using namespace TenGen::Types;

namespace TenGen::MiniJit::Instructions::Encoding {

    /**
     * @brief Generates a CBNZ (Compare and Branch on Non-Zero) instruction.
     */
    constexpr uint32_t base_br_cbnz(gpr_t Rt, int32_t imm19) {
        uint32_t ins = 0x35000000u;
        ins |= (Rt & 0x1Fu);                // Rt → bits [4:0]
        ins |= (((Rt >> 5) & 0x1u) << 31);  // sf → bit 31
        ins |= (imm19 & 0x7FFFFu) << 5;     // imm19 → bits [23:5]
        return ins;
    }

    /**
     * @brief Generates a LDP (Load Pair) instruction.
     */
    constexpr uint32_t base_ldp(gpr_t Wt1, gpr_t Wt2, gpr_t Xn_SP, uint32_t imm7) {
        uint32_t ins = 0x28C00000u;
        ins |= (Wt1 & 0x1Fu) << 0;           // Rt1 → [4:0]
        ins |= (Xn_SP & 0x1Fu) << 5;         // Rn  → [9:5]
        ins |= (Wt2 & 0x1Fu) << 10;          // Rt2 → [14:10]
        ins |= ((imm7 & 0x7Fu) << 15);       // imm7→ [21:15]
        ins |= (((Wt1 >> 5) & 0x1u) << 31);  // sf → bit 31
        return ins;
    }

    /**
     * @brief Generates a STP (Store Pair) instruction.
     */
    constexpr uint32_t base_stp(gpr_t Wt1, gpr_t Wt2, gpr_t Xn_SP, uint32_t imm7) {
        uint32_t ins = 0x28800000u;
        ins |= (Wt1 & 0x1Fu) << 0;
        ins |= (Xn_SP & 0x1Fu) << 5;
        ins |= (Wt2 & 0x1Fu) << 10;
        ins |= ((imm7 & 0x7Fu) << 15);
        ins |= (((Wt1 >> 5) & 0x1u) << 31);
        return ins;
    }

    /**
     * @brief Generates a MOV (Move Immediate) instruction using an immediate value.
     */
    constexpr uint32_t base_mov_imm(gpr_t Wd, int16_t imm16, uint8_t shift /*= 0*/) {
        uint32_t ins = 0x52800000u;       // MOVZ base for 32-bit (sf = 0)
        ins |= ((imm16 & 0xFFFFu) << 5);  // imm16 → bits [20:5]
        ins |= ((shift & 0x3u) << 21);    // shift amount (0, 16, 32, 48) → bits [22:21]
        ins |= (Wd & 0x1Fu);              // Rd → bits [4:0]
        ins |= ((Wd >> 5) & 0x1u) << 31;  // sf (bit 31): 0 = 32-bit, 1 = 64-bit
        return ins;
    }

    /**
     * @brief Generates a MOV (Move Register) instruction using a source register.
     * @param Wd  -> DST
     * @param Wm  -> SRC
     */
    constexpr uint32_t base_mov_register(gpr_t Wd, gpr_t Wm) {
        uint32_t ins = 0x2A0003E0u;  // ORR (reg) base
        ins |= (((Wd >> 5) & 0x1u) << 31);
        ins |= ((Wm & 0x1Fu) << 16);  // Rm → bits [20:16]
        ins |= (Wd & 0x1Fu);          // Rd → bits [4:0]
        return ins;
    }

    constexpr uint32_t base_movz(gpr_t Xd, uint16_t imm16, uint8_t shift /* must be 0, 16, 32, or 48 */) {
        uint32_t ins = 0xD2800000u;  // Base encoding for MOVZ

        // Set destination register Rd (bits [4:0])
        ins |= (Xd & 0x1Fu);

        // Set imm16 (bits [20:5])
        ins |= ((imm16 & 0xFFFFu) << 5);

        // Set hw (shift >> 4, bits [22:21])
        uint32_t hw = (shift / 16) & 0x3u;
        ins |= (hw << 21);

        // Set sf (bit 31) → 1 for 64-bit registers (X registers)
        ins |= ((Xd >> 5) & 0x1u) << 31;

        return ins;
    }

    constexpr uint32_t base_movk(gpr_t Xd, uint16_t imm16, uint8_t shift /* must be 0, 16, 32, or 48 */) {
        uint32_t ins = 0xF2800000u;  // Base encoding for MOVK

        // Set destination register Rd (bits [4:0])
        ins |= (Xd & 0x1Fu);

        // Set imm16 (bits [20:5])
        ins |= ((imm16 & 0xFFFFu) << 5);

        // Set hw (shift >> 4, bits [22:21])
        uint32_t hw = (shift / 16) & 0x3u;
        ins |= (hw << 21);

        // Set sf (bit 31) → 1 for 64-bit registers (X registers)
        ins |= ((Xd >> 5) & 0x1u) << 31;

        return ins;
    }

    /**
     * @brief Generates an ADD (Add Immediate) instruction.
     */
    constexpr uint32_t base_add_imm(gpr_t Wd, gpr_t Wn, int32_t imm12, int32_t shift) {
        uint32_t ins = 0x11000000u;
        ins |= (((Wd >> 5) & 0x1u) << 31);
        ins |= ((shift & 1u) << 22);  // LSL #shift? only 0 or 1
        ins |= (imm12 & 0xFFFu) << 10;
        ins |= ((Wn & 0x1Fu) << 5);
        ins |= (Wd & 0x1Fu);
        return ins;
    }

    /** @brief
     *
     */
    constexpr uint32_t base_add_shifted_register(gpr_t Wd, gpr_t Wn, gpr_t Wm, int32_t shift_type, uint32_t imm6) {
        uint32_t ins = 0x0B000000u;
        ins |= (((Wd >> 5) & 0x1u) << 31);
        ins |= ((shift_type & 0x3u) << 22);
        ins |= ((imm6 & 0x3Fu) << 10);
        ins |= ((Wm & 0x1Fu) << 16);
        ins |= ((Wn & 0x1Fu) << 5);
        ins |= (Wd & 0x1Fu);
        return ins;
    }

    /**
     * @brief Generates a SUB (Subtract Immediate) instruction.
     */
    constexpr uint32_t base_sub_imm(gpr_t Wd, gpr_t Wn, int32_t imm12, int32_t shift) {
        uint32_t ins = 0x51000000u;
        ins |= (((Wd >> 5) & 0x1u) << 31);
        ins |= ((shift & 1u) << 22);
        ins |= (imm12 & 0xFFFu) << 10;
        ins |= ((Wn & 0x1Fu) << 5);
        ins |= (Wd & 0x1Fu);
        return ins;
    }

    /**
     * @brief Generates a SUB (Subtract Shifted Register) instruction.
     */
    constexpr uint32_t base_sub_shifted_register(gpr_t Wd, gpr_t Wn, gpr_t Wm, uint32_t shift_type, uint32_t imm6) {
        uint32_t ins = 0x4B000000u;
        ins |= (((Wd >> 5) & 0x1u) << 31);
        ins |= ((shift_type & 0x3u) << 22);
        ins |= ((imm6 & 0x3Fu) << 10);
        ins |= ((Wm & 0x1Fu) << 16);
        ins |= ((Wn & 0x1Fu) << 5);
        ins |= (Wd & 0x1Fu);
        return ins;
    }

    /**
     * @brief Generates a LSL (Logical Shift Left Immediate) instruction.
     */
    constexpr uint32_t base_lsl_imm(gpr_t Wd, gpr_t Wn, uint32_t shift) {
        // LSL #n is an alias for UBFM Rd, Rn, #(-n & 31), #(31 - n)
        uint32_t immr = (-shift) & 0x1F;
        uint32_t imms = 31 - shift;

        uint32_t ins = 0x53000000u;
        ins |= (immr & 0x3F) << 16;
        ins |= (imms & 0x3F) << 10;
        ins |= (Wn & 0x1F) << 5;
        ins |= (Wd & 0x1F);
        return ins;
    }

    /**
     * @brief Generates a LSL (Logical Shift Left Shifted Register) instruction.
     */
    constexpr uint32_t base_lsl_register(gpr_t Wd, gpr_t Wn, gpr_t Wm) {
        uint32_t ins = 0x1AC02000u;
        ins |= (((Wd >> 5) & 0x1u) << 31);
        ins |= ((Wm & 0x1Fu) << 16);
        ins |= ((Wn & 0x1Fu) << 5);
        ins |= (Wd & 0x1Fu);
        return ins;
    }

    /**
     * @brief Generates a Mul instruction ( Rd = Rn * Rm )
     */
    constexpr uint32_t base_mul_reg(gpr_t dst, gpr_t src_1, gpr_t src_0) {
        uint32_t l_ins = 0x1b007c00;
        l_ins |= (0x1f & dst);
        l_ins |= (0x1f & src_1) << 5;
        l_ins |= (0x1f & src_0) << 16;
        l_ins |= (0x20 & dst) << 26;
        return l_ins;
    }

    /**
     * @brief Generates a RET (Return from Subroutine) instruction.
     */
    constexpr uint32_t base_ret() {
        return 0xd65f03c0;
    }

    /**
     * @brief Generates an FMLA (vector) instruction.
     *
     * @param reg_dest destination register.
     * @param reg_src1 first source register.
     * @param reg_src2 second source register.
     * @param arr_spec arrangement specifier.
     *
     * @return instruction.
     **/
    constexpr uint32_t neon_fmla_vector(simd_fp_t reg_dest,
                                        simd_fp_t reg_src1,
                                        simd_fp_t reg_src2,
                                        arr_spec_t arr_spec) {
        uint32_t l_ins = 0x0e20cc00;

        // set destination register id
        uint32_t l_reg_id = reg_dest & 0x1f;
        l_ins |= l_reg_id;

        // set first source register id
        l_reg_id = reg_src1 & 0x1f;
        l_ins |= l_reg_id << 5;

        // set second source register id
        l_reg_id = reg_src2 & 0x1f;
        l_ins |= l_reg_id << 16;

        // set arrangement specifier
        uint32_t l_arr_spec = arr_spec & 0x40400000;
        l_ins |= l_arr_spec;

        return l_ins;
    }

    /**
     * @brief Generates an FMLA (element) instruction.
     *
     * @param reg_dest destination register.
     * @param reg_src1 first source register.
     * @param reg_src2 second source register.
     * @param element_spec precision and element specifier.
     *
     * @return instruction.
     **/
    constexpr uint32_t neon_fmla_element(simd_fp_t reg_dest,
                                         simd_fp_t reg_src1,
                                         simd_fp_t reg_src2,
                                         element_spec_t element_spec) {
        uint32_t l_ins = 0x0f801000;

        // set destination register id
        uint32_t l_reg_id = reg_dest & 0x1f;
        l_ins |= l_reg_id;

        // set first source register id
        l_reg_id = reg_src1 & 0x1f;
        l_ins |= l_reg_id << 5;

        // set second source register id
        l_reg_id = reg_src2 & 0x1f;
        l_ins |= l_reg_id << 16;

        // set element specifier
        uint32_t l_arr_spec = element_spec & 0x40700800;
        l_ins |= l_arr_spec;

        return l_ins;
    }

    constexpr uint32_t neon_ldr(simd_fp_t reg_dst,
                                gpr_t add_src,
                                int32_t imm9,
                                arr_spec_t i_dtype) {
        uint32_t l_inst = 0x3c40'0400;

        l_inst |= (reg_dst & 0x1f);
        l_inst |= i_dtype;
        l_inst |= (add_src & 0x1f) << 5;
        l_inst |= (imm9 % 0x1ff) << 12;

        return l_inst;
    }

    constexpr uint32_t neon_str(simd_fp_t reg_dst,
                                gpr_t add_src,
                                int32_t imm9,
                                arr_spec_t i_dtype) {
        uint32_t l_inst = 0xfc00'0400;

        l_inst |= (reg_dst & 0x1f);
        l_inst |= i_dtype;
        l_inst |= (add_src & 0x1f) << 5;
        l_inst |= (imm9 % 0x1ff) << 12;

        return l_inst;
    }

    constexpr uint32_t neon_ld1_no_offset(simd_fp_t reg_dst,
                                          gpr_t add_src,
                                          vector_count_t reg_count) {
        uint32_t l_inst = 0x4c402000;

        l_inst |= (reg_dst & 0x1F);       // Rt: bits 4:0
        l_inst |= (add_src & 0x1F) << 5;  // Rn: bits 9:5
        l_inst |= (0b10) << 10;           // size: bits 11:10
        l_inst |= reg_count;              // opcode: bits 15:12
        l_inst |= 0b1 << 30;              // Q (size): 31

        return l_inst;
    }

    /**
     * @brief Generates an LD1 (single structure) instruction.
     *
     * @param reg_dest destination register.
     * @param reg_src  source register.
     * @param lane_index element specifier (0 to 3)
     *
     * @return instruction.
     **/
    constexpr uint32_t neon_ld1_scalar_index(simd_fp_t reg_dst,
                                             gpr_t reg_src,
                                             uint32_t index) {
        uint32_t l_inst = 0xD408000;

        uint32_t q = (index >> 1) & 0x1;  // Q
        uint32_t s = index & 0x1;         // S

        l_inst |= (reg_dst & 0x1Fu) << 0;  // Rt: bits 4:0
        l_inst |= (reg_src & 0x1Fu) << 5;  // Rn: bits 9:5
        l_inst |= (q << 30) | (s << 12);

        return l_inst;
    }

    constexpr uint32_t neon_st1_no_offset(simd_fp_t reg_dst,
                                          gpr_t add_src,
                                          vector_count_t reg_count) {
        uint32_t l_inst = 0x0C00'2000;

        l_inst |= (reg_dst & 0x1F);       // Rt: bits 4:0
        l_inst |= (add_src & 0x1F) << 5;  // Rn: bits 9:5
        l_inst |= (0b10) << 10;           // size: bits 11:10
        l_inst |= reg_count;              // opcode: bits 15:12
        l_inst |= 0b1 << 30;              // Q (size): 31

        return l_inst;
    }

    /**
     * @brief Generates an ST1 (single structure) instruction.
     *
     * @param reg_dest destination register.
     * @param reg_src  source register.
     * @param lane_index element specifier (0 to 3)
     *
     * @return instruction.
     **/
    constexpr uint32_t neon_st1_scalar_index(simd_fp_t reg_dst,
                                             gpr_t reg_src,
                                             uint32_t index) {
        uint32_t l_inst = 0xD008000;

        uint32_t q = (index >> 1) & 0x1;  // Q
        uint32_t s = index & 0x1;         // S

        l_inst |= (reg_dst & 0x1Fu) << 0;  // Rt: bits 4:0
        l_inst |= (reg_src & 0x1Fu) << 5;  // Rn: bits 9:5
        l_inst |= (q << 30) | (s << 12);

        return l_inst;
    }

    constexpr uint32_t neon_fmla_by_element(simd_fp_t reg_dest,
                                            simd_fp_t reg_src1,
                                            simd_fp_t reg_src2,
                                            uint32_t arr_index) {
        uint32_t l_ins = 0xfc00'0400;

        l_ins |= (reg_dest & 0x1f);
        l_ins |= (reg_src1 & 0x1f) << 5;
        l_ins |= (reg_src2 & 0x1f) << 16;

        l_ins |= (arr_index & 0x1) << 11;
        l_ins |= (arr_index & 0x2) << 21;

        return l_ins;
    }

    constexpr uint32_t neon_movi_zero(simd_fp_t reg_dest,
                                      bool use_full_register,
                                      bool use_double) {
        // MOVI base encoding
        uint32_t l_ins = 0x0f000400;

        // Set Q bit
        if (use_full_register) {
            l_ins |= (1u << 30);
        }

        uint8_t size_field = 0b10;  // default 32-bit (float)
        if (use_double) {
            size_field = 0b11;  // 64-bit (double)
        }
        l_ins |= (size_field << 29);

        // Destination register (bits 4:0)
        l_ins |= (reg_dest & 0x1f);

        return l_ins;
    }

    constexpr uint32_t neon_fmaxnmp_vector(simd_fp_t reg_dest,
                                           simd_fp_t reg_src1,
                                           simd_fp_t reg_src2,
                                           bool is_double_precision) {
        // Base encoding for FMAXNMP (vector), per ARMv8-A spec
        uint32_t l_ins = 0x2e20c400;  // FMAXNMP vector base opcode

        // Set Q = 1 (128-bit vector)
        l_ins |= (1 << 30);

        // Set sz bit based on precision: 0 for 32-bit, 1 for 64-bit
        if (is_double_precision)
            l_ins |= (1 << 22);  // sz = 1

        // Set Rm (source register 2)
        uint32_t l_reg_id = reg_src2 & 0x1f;
        l_ins |= l_reg_id << 16;

        // Set Rn (source register 1)
        l_reg_id = reg_src1 & 0x1f;
        l_ins |= l_reg_id << 5;

        // Set Rd (destination register)
        l_reg_id = reg_dest & 0x1f;
        l_ins |= l_reg_id;

        return l_ins;
    }

    constexpr uint32_t neon_fmax_vector(simd_fp_t reg_dest,
                                        simd_fp_t reg_src1,
                                        simd_fp_t reg_src2,
                                        bool is_double_precision) {
        // Base encoding for FMAX (vector), element-wise
        uint32_t l_ins = 0x0e20f400;

        // Set Q = 1 (128-bit vector)
        l_ins |= (1 << 30);

        // Set sz bit (bit 22): 0 for 32-bit (single), 1 for 64-bit (double)
        if (is_double_precision)
            l_ins |= (1 << 22);

        // Set Rm (source register 2) bits [20:16]
        l_ins |= (reg_src2 & 0x1f) << 16;

        // Set Rn (source register 1) bits [9:5]
        l_ins |= (reg_src1 & 0x1f) << 5;

        // Set Rd (destination register) bits [4:0]
        l_ins |= (reg_dest & 0x1f);

        return l_ins;
    }

    constexpr uint32_t neon_ld1_multiple(simd_fp_t base_reg,
                                         gpr_t src_reg,
                                         ld1_opcode_t op_code,
                                         ld1_t element) {
        // Basis-Opcode für LD1 (multiple structures)
        uint32_t l_ins = 0x0c402000;

        l_ins |= element;
        l_ins |= op_code << 12;

        // Set Rn (source register) bits [9:5]
        l_ins |= (src_reg & 0x1f) << 5;

        // Set Rt (first or only register) bits [4:0]
        l_ins |= (base_reg & 0x1f);

        return l_ins;
    }

    constexpr uint32_t neon_st1_multiple(simd_fp_t base_reg,
                                         gpr_t src_reg,
                                         ld1_opcode_t op_code,
                                         ld1_t element) {
        // Basis-Opcode für LD1 (multiple structures)
        uint32_t l_ins = 0x0c002000;

        l_ins |= element;
        l_ins |= op_code << 12;

        // Set Rn (source register) bits [9:5]
        l_ins |= (src_reg & 0x1f) << 5;

        // Set Rt (first or only register) bits [4:0]
        l_ins |= (base_reg & 0x1f);

        return l_ins;
    }

}  // namespace TenGen::MiniJit::Instructions::Encoding

#endif  // TENGEN_MINI_JIT_INSTRUCTIONS_ENCODING_H