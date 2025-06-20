#ifndef MINI_JIT_INSTRUCTIONS_INSTRUCTIONS_H
#define MINI_JIT_INSTRUCTIONS_INSTRUCTIONS_H

#include <math.h>

#include <cstdint>
#include <string>

namespace mini_jit::instructions {
    class InstGen;
}

class mini_jit::instructions::InstGen {
   public:
    //! general-purpose registers
    typedef enum : uint32_t {
        w0 = 0,
        w1 = 1,
        w2 = 2,
        w3 = 3,
        w4 = 4,
        w5 = 5,
        w6 = 6,
        w7 = 7,
        w8 = 8,
        w9 = 9,
        w10 = 10,
        w11 = 11,
        w12 = 12,
        w13 = 13,
        w14 = 14,
        w15 = 15,
        w16 = 16,
        w17 = 17,
        w18 = 18,
        w19 = 19,
        w20 = 20,
        w21 = 21,
        w22 = 22,
        w23 = 23,
        w24 = 24,
        w25 = 25,
        w26 = 26,
        w27 = 27,
        w28 = 28,
        w29 = 29,
        w30 = 30,

        x0 = 32 + 0,
        x1 = 32 + 1,
        x2 = 32 + 2,
        x3 = 32 + 3,
        x4 = 32 + 4,
        x5 = 32 + 5,
        x6 = 32 + 6,
        x7 = 32 + 7,
        x8 = 32 + 8,
        x9 = 32 + 9,
        x10 = 32 + 10,
        x11 = 32 + 11,
        x12 = 32 + 12,
        x13 = 32 + 13,
        x14 = 32 + 14,
        x15 = 32 + 15,
        x16 = 32 + 16,
        x17 = 32 + 17,
        x18 = 32 + 18,
        x19 = 32 + 19,
        x20 = 32 + 20,
        x21 = 32 + 21,
        x22 = 32 + 22,
        x23 = 32 + 23,
        x24 = 32 + 24,
        x25 = 32 + 25,
        x26 = 32 + 26,
        x27 = 32 + 27,
        x28 = 32 + 28,
        x29 = 32 + 29,
        x30 = 32 + 30,

        wzr = 31,
        xzr = 32 + 31,
        sp = 64 + 32 + 31
    } gpr_t;

    //! simd&fp registers
    typedef enum : uint32_t {
        v0 = 0,
        v1 = 1,
        v2 = 2,
        v3 = 3,
        v4 = 4,
        v5 = 5,
        v6 = 6,
        v7 = 7,
        v8 = 8,
        v9 = 9,
        v10 = 10,
        v11 = 11,
        v12 = 12,
        v13 = 13,
        v14 = 14,
        v15 = 15,
        v16 = 16,
        v17 = 17,
        v18 = 18,
        v19 = 19,
        v20 = 20,
        v21 = 21,
        v22 = 22,
        v23 = 23,
        v24 = 24,
        v25 = 25,
        v26 = 26,
        v27 = 27,
        v28 = 28,
        v29 = 29,
        v30 = 30,
        v31 = 31
    } simd_fp_t;

    //! arrangement specifiers
    typedef enum : uint32_t {
        b = 0x0,
        h = 0x40000000,
        s = 0x80000000,
        d = 0xc0000000,
        q = 0x00800000,
    } arr_spec_t;

    typedef enum : uint32_t {
        S2_0 = 0x00000000,
        S2_1 = 0x00200000,
        S2_2 = 0x00000800,
        S2_3 = 0x00200800,
        S4_0 = 0x40000000,
        S4_1 = 0x40200000,
        S4_2 = 0x40000800,
        S4_3 = 0x40200800,
        D2_0 = 0x40400000,
        D2_1 = 0x40400800
    } element_spec_t;

    typedef enum : uint32_t {
        S2 = 0x800,
        S4 = 0x40000800,
        D2 = 0x40000c00
    } ld1_t;

    typedef enum : uint32_t {
        vc1 = 0x5800,
        vc2 = 0x8800,
        vc3 = 0x4800,
        vc4 = 0x800,
    } vector_count_t;

    typedef enum : uint32_t {
        one_regs = 0x7,
        two_regs = 0xa,
        three_regs = 0x6,
        four_regs = 0x2,
    } ld1_opcode_t;

    /**
     * @brief Generates a CBNZ (Compare and Branch on Non-Zero) instruction.
     */
    static uint32_t base_br_cbnz(gpr_t reg, int32_t imm19);

    /**
     * @brief Generates a LDP (Load Pair) instruction.
     */
    static uint32_t base_ldp(gpr_t Wt1, gpr_t Wt2, gpr_t Xn_SP, uint32_t imm7);

    /**
     * @brief Generates a STP (Store Pair) instruction.
     */
    static uint32_t base_stp(gpr_t Wt1, gpr_t Wt2, gpr_t Xn_SP, uint32_t imm7);

    /**
     * @brief Generates a MOV (Move Immediate) instruction using an immediate value.
     */
    static uint32_t base_mov_imm(gpr_t Wd, int16_t imm16, uint8_t shift /*= 0*/);

    /**
     * @brief Generates a MOV (Move Register) instruction using a source register.
     * @param Wd  -> DST
     * @param Wm  -> SRC
     */
    static uint32_t base_mov_register(gpr_t dst_reg, gpr_t src_reg);

    static uint32_t base_movz(gpr_t Xd, uint16_t imm16, uint8_t shift /* must be 0, 16, 32, or 48 */);

    static uint32_t base_movk(gpr_t Xd, uint16_t imm16, uint8_t shift /* must be 0, 16, 32, or 48 */);

    /**
     * @brief Generates an ADD (Add Immediate) instruction.
     */
    static uint32_t base_add_imm(gpr_t Wd_WSP, gpr_t Wn_WSP, int32_t imm12, int32_t shift);

    /** @brief
     *
     */
    static uint32_t base_add_shifted_register(gpr_t Wd, gpr_t Wn, gpr_t Wm, int32_t shift_type, uint32_t imm6);

    /**
     * @brief Generates a SUB (Subtract Immediate) instruction.
     */
    static uint32_t base_sub_imm(gpr_t Wd_WSP, gpr_t Wn_WSP, int32_t imm12, int32_t shift);

    /**
     * @brief Generates a SUB (Subtract Shifted Register) instruction.
     */
    static uint32_t base_sub_shifted_register(gpr_t Wd, gpr_t Wn, gpr_t Wm, uint32_t shift_type, uint32_t imm6);

    /**
     * @brief Generates a LSL (Logical Shift Left Immediate) instruction.
     */
    static uint32_t base_lsl_imm(gpr_t Wd, gpr_t Wn, uint32_t shift);

    /**
     * @brief Generates a LSL (Logical Shift Left Shifted Register) instruction.
     */
    static uint32_t base_lsl_register(gpr_t Wd, gpr_t Wn, gpr_t Wm);

    /**
     * @brief Generates a Mul instruction ( Rd = Rn * Rm )
     */
    static uint32_t base_mul_reg(gpr_t dst, gpr_t src_1, gpr_t src_0);

    /**
     * @brief Generates a RET (Return from Subroutine) instruction.
     */
    static uint32_t base_ret();

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
    static uint32_t neon_fmla_vector(simd_fp_t reg_dest,
                                     simd_fp_t reg_src1,
                                     simd_fp_t reg_src2,
                                     arr_spec_t arr_spec);

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
    static uint32_t neon_fmla_element(simd_fp_t reg_dest,
                                      simd_fp_t reg_src1,
                                      simd_fp_t reg_src2,
                                      element_spec_t element_spec);

    static uint32_t neon_ldr(simd_fp_t reg_dst,
                             gpr_t add_src,
                             int32_t imm9,
                             arr_spec_t i_dtype);

    static uint32_t neon_str(simd_fp_t reg_dst,
                             gpr_t add_src,
                             int32_t imm9,
                             arr_spec_t i_dtype);

    static uint32_t neon_ld1_no_offset(simd_fp_t reg_dst,
                                       gpr_t add_src,
                                       vector_count_t reg_count);

    /**
     * @brief Generates an LD1 (single structure) instruction.
     *
     * @param reg_dest destination register.
     * @param reg_src  source register.
     * @param lane_index element specifier (0 to 3)
     *
     * @return instruction.
     **/
    static uint32_t neon_ld1_scalar_index(simd_fp_t reg_dst, gpr_t reg_src, int index);

    /**
     * @brief Generates an ST1 (single structure) instruction.
     *
     * @param reg_dest destination register.
     * @param reg_src  source register.
     * @param lane_index element specifier (0 to 3)
     *
     * @return instruction.
     **/
    static uint32_t neon_st1_scalar_index(simd_fp_t reg_dst, gpr_t reg_src, int index);

    static uint32_t neon_fmla_by_element(simd_fp_t reg_dest,
                                         simd_fp_t reg_src1,
                                         simd_fp_t reg_src2,
                                         uint32_t arr_index);

    static uint32_t neon_st1_no_offset(simd_fp_t reg_dst,
                                       gpr_t add_src,
                                       vector_count_t reg_count);

    static uint32_t neon_movi_zero(simd_fp_t reg_dest,
                                   bool use_full_register,
                                   bool use_double);

    static uint32_t neon_fmaxnmp_vector(simd_fp_t reg_dest,
                                        simd_fp_t reg_src1,
                                        simd_fp_t reg_src2,
                                        bool is_double_precision);

    static uint32_t neon_fmax_vector(simd_fp_t reg_dest,
                                     simd_fp_t reg_src1,
                                     simd_fp_t reg_src2,
                                     bool is_double_precision);

    static uint32_t neon_ld1_multiple(simd_fp_t base_reg,
                                      gpr_t src_reg,
                                      ld1_opcode_t op_code,
                                      ld1_t element);

    static uint32_t neon_st1_multiple(simd_fp_t base_reg,
                                      gpr_t src_reg,
                                      ld1_opcode_t op_code,
                                      ld1_t element);

    static uint32_t neon_trn( simd_fp_t reg_dst,
                              simd_fp_t reg_src1,
                              simd_fp_t reg_src2,
                              int variant /* 1 or 2 */);
    static uint32_t neon_zip( simd_fp_t reg_dst,
                              simd_fp_t reg_src1,
                              simd_fp_t reg_src2,
                              int variant /* 1 or 2 */);
    static uint32_t neon_eor( simd_fp_t reg_dst,
                              simd_fp_t reg_src1,
                              simd_fp_t reg_src2 );
};
#endif