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

uint32_t mini_jit::instructions::InstGen::neon_ldr(simd_fp_t reg_dst,
                                                   gpr_t add_src,
                                                   int32_t imm9) {
    uint32_t l_inst = 0xbd400000;

    l_inst |= (reg_dst & 0x1f);
    l_inst |= (add_src & 0x1f) << 5;
    l_inst |= (imm9 % 0x1ff) << 12;

    return l_inst;
}