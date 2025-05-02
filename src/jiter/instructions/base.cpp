#include "instructions.h"

uint32_t jiter::instructions::InstGen::base_br_cbnz(gpr_t reg,
                                                    int32_t imm19) {
    uint32_t l_ins = 0x35000000;

    // set register id
    uint32_t l_reg_id = reg & 0x1f;
    l_ins |= l_reg_id;

    // set size of the register
    uint32_t l_reg_size = reg & 0x20;
    l_ins |= l_reg_size << (32 - 6);

    // set immediate
    uint32_t l_imm = imm19 & 0x7ffff;
    l_ins |= l_imm << 5;

    return l_ins;
}