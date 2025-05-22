#include <cstring>
#include <iostream>

#include "instructions.h"

uint32_t mini_jit::instructions::InstGen::neon_fmla_vector(simd_fp_t reg_dest,
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

uint32_t mini_jit::instructions::InstGen::neon_fmla_element(simd_fp_t reg_dest,
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

uint32_t mini_jit::instructions::InstGen::neon_ldr(simd_fp_t reg_dst,
                                                   gpr_t add_src,
                                                   int32_t imm9) {
    uint32_t l_inst = 0xbd400000;

    l_inst |= (reg_dst & 0x1f);
    l_inst |= (add_src & 0x1f) << 5;
    l_inst |= (imm9 % 0x1ff) << 12;

    return l_inst;
}

uint32_t mini_jit::instructions::InstGen::neon_fmla_by_element(simd_fp_t reg_dest,
                                                               simd_fp_t reg_src1,
                                                               simd_fp_t reg_src2,
                                                               uint32_t arr_index) {
    uint32_t l_ins = 0x4f801000;

    l_ins |= (reg_dest & 0x1f);
    l_ins |= (reg_src1 & 0x1f) << 5;
    l_ins |= (reg_src2 & 0x1f) << 16;

    l_ins |= (arr_index & 0x1) << 11;
    l_ins |= (arr_index & 0x2) << 21;

    return l_ins;
}

// TODO: MAKE THIS DYNAMIC FOR MORE V REGISTERS AND OTHER SIZES THEN 4s.

/*
 * LD1 instruction for ONLY one v register at a time with 4s.
 */
uint32_t mini_jit::instructions::InstGen::neon_ld1_no_offset(simd_fp_t reg_dst,
                                                             gpr_t add_src,
                                                             vector_count_t /*reg_count*/) {
    uint32_t l_inst = 0xC400000;

    l_inst |= (reg_dst & 0x1F);       // Rt: bits 4:0
    l_inst |= (add_src & 0x1F) << 5;  // Rn: bits 9:5
    l_inst |= (0b10) << 10;           // size: bits 11:10
    l_inst |= 0b0111 << 12;           // opcode: bits 15:12
    l_inst |= 0b1 << 30;              // Q (size): 31

    return l_inst;
}

// LD1 { <Vt>.S }[<index>], [<Xn|SP>]
uint32_t mini_jit::instructions::InstGen::neon_ld1_scalar_index(simd_fp_t reg_dst,
                                                                gpr_t reg_src,
                                                                int index) {
    uint32_t l_inst = 0xD408000;

    uint32_t q = (index >> 1) & 0x1;  // Q
    uint32_t s = index & 0x1;         // S

    l_inst |= (reg_dst & 0x1Fu) << 0;  // Rt: bits 4:0
    l_inst |= (reg_src & 0x1Fu) << 5;  // Rn: bits 9:5
    l_inst |= (q << 30) | (s << 12);

    return l_inst;
}

// TODO: MAKE THIS DYNAMIC FOR MORE V REGISTERS AND OTHER SIZES THEN 4s.

/*
 * ST1 instruction for ONLY one v register at a time with 4s.
 */
uint32_t mini_jit::instructions::InstGen::neon_st1_no_offset(simd_fp_t reg_dst,
                                                             gpr_t add_src,
                                                             vector_count_t /*reg_count*/) {
    uint32_t l_inst = 0xC000000;

    l_inst |= (reg_dst & 0x1F);       // Rt: bits 4:0
    l_inst |= (add_src & 0x1F) << 5;  // Rn: bits 9:5
    l_inst |= (0b10) << 10;           // size: bits 11:10
    l_inst |= 0b0111 << 12;           // opcode: bits 15:12
    l_inst |= 0b1 << 30;              // Q (size): 31

    return l_inst;
}

uint32_t mini_jit::instructions::InstGen::neon_st1_scalar_index(simd_fp_t reg_dst,
                                                                gpr_t reg_src,
                                                                int index) {
    uint32_t l_inst = 0xD008000;

    uint32_t q = (index >> 1) & 0x1;  // Q
    uint32_t s = index & 0x1;         // S

    l_inst |= (reg_dst & 0x1Fu) << 0;  // Rt: bits 4:0
    l_inst |= (reg_src & 0x1Fu) << 5;  // Rn: bits 9:5
    l_inst |= (q << 30) | (s << 12);

    return l_inst;
}

uint32_t mini_jit::instructions::InstGen::neon_movi_zero(simd_fp_t reg_dest,
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

uint32_t mini_jit::instructions::InstGen::neon_fmaxnmp_vector(simd_fp_t reg_dest,
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

uint32_t mini_jit::instructions::InstGen::neon_fmax_vector(simd_fp_t reg_dest,
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
