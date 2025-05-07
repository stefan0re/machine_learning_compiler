#include "instructions.h"

uint32_t jiter::instructions::InstGen::neon_fmla_vector(simd_fp_t reg_dest,
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

uint32_t jiter::instructions::InstGen::neon_fmla_element(simd_fp_t reg_dest,
                                                         simd_fp_t reg_src1,
                                                         simd_fp_t reg_src2,
                                                         element_spec_t element_spec) {
    uint32_t l_ins = 0xBF001000;

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
    uint32_t l_arr_spec = element_spec & 0x00700000;
    l_ins |= l_arr_spec;

    return l_ins;
}