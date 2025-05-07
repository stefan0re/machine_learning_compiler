#include "instructions.h"

uint32_t jiter::instructions::InstGen::base_br_cbnz(gpr_t reg, int32_t imm19) {
    // 32 bit optcode for cbnz without variables
    uint32_t l_ins = 0x35000000;  // 00110101000000000000000000000000

    // keeping only the lowest 5 bits of reg, which represent the register number (0–31).
    uint32_t l_reg_id = reg & 0x1f;  // reg &  00011111 (rest is leading zeros, so 8 bit are enough)
    l_ins |= l_reg_id;

    // determin w or x view:
    // shifting 26 places left (from bit 5 → bit 31), so it ends up in the
    // correct size field of the instruction (bit 31):
    // If reg = 0x21 (binary: 00100001), then:
    //      l_reg_id = 0x01 → Register 1
    //      l_reg_size = 0x20 → This gets shifted into bit 31 → sets size = 1 (64-bit register)
    uint32_t l_reg_size = reg & 0x20;  // reg & 00100000
    l_ins |= l_reg_size << (32 - 6);

    // set immediate (address of the jmp):
    // cast the 19 bit address to 32 bit
    uint32_t l_imm = imm19 & 0x7ffff;  // 01111111111111111111
    // 19-bit immediate goes into bits [23:5] of the 32-bit instruction
    l_ins |= l_imm << 5;

    return l_ins;
}

uint32_t jiter::instructions::InstGen::base_ldp(gpr_t Wt1, gpr_t Wt2, gpr_t Xn_SP, int32_t imm7) {
    uint32_t l_ins = 0x28C00000;

    // extract lower 5 bits from each and shift them to correct position
    uint32_t wt1_bits = (Wt1 & 0x1F) << 0;
    uint32_t xnsp_bits = (Xn_SP & 0x1F) << 5;
    uint32_t wt2_bits = (Wt2 & 0x1F) << 10;
    uint32_t imm7_bits = ((imm7 & 0x7F) << 15);

    // insert the new bits
    l_ins |= wt1_bits | xnsp_bits | wt2_bits | imm7_bits;

    // determin w or x view:
    uint32_t l_reg_size = Wt1 & 0x20;  // reg & 00100000
    l_ins |= l_reg_size << (32 - 6);

    return l_ins;
}

uint32_t jiter::instructions::InstGen::base_stp(gpr_t Wt1, gpr_t Wt2, gpr_t Xn_SP, int32_t imm7) {
    uint32_t l_ins = 0x28800000;

    // extract lower 5 bits from each and shift them to correct position
    uint32_t wt1_bits = (Wt1 & 0x1F) << 0;
    uint32_t xnsp_bits = (Xn_SP & 0x1F) << 5;
    uint32_t wt2_bits = (Wt2 & 0x1F) << 10;
    uint32_t imm7_bits = ((imm7 & 0x7F) << 15);

    // insert the new bits
    l_ins |= wt1_bits | xnsp_bits | wt2_bits | imm7_bits;

    // determin w or x view:
    uint32_t l_reg_size = Wt1 & 0x20;  // reg & 00100000
    l_ins |= l_reg_size << (32 - 6);

    return l_ins;
}

uint32_t jiter::instructions::InstGen::base_mov_imm(gpr_t Wd_WSP, int32_t imm12) {
    // here the opc AND the Rn field is encoded
    uint32_t l_ins = 0x288003E0;

    return l_ins;
}

uint32_t jiter::instructions::InstGen::base_mov_register(gpr_t Wd, gpr_t Wm) {
    uint32_t l_ins = 0x2A0003E0;

    // extract lower 5 bits from each and shift them to correct position
    uint32_t wd_bits = (Wd & 0x1F) << 0;
    uint32_t wm_bits = (Wm & 0x1F) << 15;

    // insert the new bits
    l_ins |= wm_bits | wd_bits;

    // determin w or x view:
    uint32_t l_reg_size = Wd & 0x20;
    l_ins |= l_reg_size << (32 - 6);

    return l_ins;
}

uint32_t jiter::instructions::InstGen::base_add_imm(gpr_t Wd_WSP, gpr_t Wn_WSP, int32_t imm12, int32_t shift) {
    uint32_t l_ins = 0x11000000;

    // extract lower 5 bits from each and shift them to correct position
    uint32_t wd_wsp_bits = (Wd_WSP & 0x1F) << 0;
    uint32_t wn_wsp_bits = (Wn_WSP & 0x1F) << 5;
    uint32_t imm12_bits = (imm12 & 0xFFF) << 10;
    uint32_t sh_bit = (shift & 0x01) << 22;

    // insert the new bits
    l_ins |= sh_bit | imm12_bits | wn_wsp_bits | wd_wsp_bits;

    // determin w or x view:
    uint32_t l_reg_size = Wd_WSP & 0x20;
    l_ins |= l_reg_size << (32 - 6);

    return l_ins;
}

uint32_t jiter::instructions::InstGen::base_add_shifted_register(gpr_t Wd, gpr_t Wn, gpr_t Wm, uint32_t shift_type, uint32_t imm6) {
    uint32_t l_ins = 0x0B000000;

    // extract lower 5 bits from each and shift them to correct position
    uint32_t wd_bits = (Wd & 0x1F) << 0;
    uint32_t wn_bits = (Wn & 0x1F) << 5;
    uint32_t imm6_bits = (imm6 & 0x3F) << 10;
    uint32_t wm_bits = (Wm & 0x1F) << 16;
    uint32_t sh_bit = (shift_type & 0x03) << 22;

    // insert the new bits
    l_ins |= sh_bit | wm_bits | imm6_bits | wn_bits | wd_bits;

    // determin w or x view:
    uint32_t l_reg_size = Wd & 0x20;
    l_ins |= l_reg_size << (32 - 6);

    return l_ins;
}

uint32_t jiter::instructions::InstGen::base_sub_imm(gpr_t Wd_WSP, gpr_t Wn_WSP, int32_t imm12, int32_t shift) {
    uint32_t l_ins = 0x51000000;

    // extract lower 5 bits from each and shift them to correct position
    uint32_t wd_wsp_bits = (Wd_WSP & 0x1F) << 0;
    uint32_t wn_wsp_bits = (Wn_WSP & 0x1F) << 5;
    uint32_t imm12_bits = (imm12 & 0xFFF) << 10;
    uint32_t sh_bit = (shift & 0x01) << 22;

    // insert the new bits
    l_ins |= sh_bit | imm12_bits | wn_wsp_bits | wd_wsp_bits;

    // determin w or x view:
    uint32_t l_reg_size = Wd_WSP & 0x20;
    l_ins |= l_reg_size << (32 - 6);

    return l_ins;
}

uint32_t jiter::instructions::InstGen::base_sub_shifted_register(gpr_t Wd, gpr_t Wn, gpr_t Wm, uint32_t shift_type, uint32_t imm6) {
    uint32_t l_ins = 0x4B000000;

    // extract lower 5 bits from each and shift them to correct position
    uint32_t wd_bits = (Wd & 0x1F) << 0;
    uint32_t wn_bits = (Wn & 0x1F) << 5;
    uint32_t imm6_bits = (imm6 & 0x3F) << 10;
    uint32_t wm_bits = (Wm & 0x1F) << 16;
    uint32_t sh_bit = (shift_type & 0x03) << 22;

    // insert the new bits
    l_ins |= sh_bit | wm_bits | imm6_bits | wn_bits | wd_bits;

    // determin w or x view:
    uint32_t l_reg_size = Wd & 0x20;
    l_ins |= l_reg_size << (32 - 6);

    return l_ins;
}

uint32_t jiter::instructions::InstGen::base_lsl_imm(gpr_t Wd, gpr_t Wn, uint32_t shift) {
    uint32_t l_ins = 0x53000000;

    // extract lower 5 bits from each and shift them to correct position
    uint32_t wd_bits = (Wd & 0x1F) << 0;
    uint32_t wn_bits = (Wn & 0x1F) << 5;
    uint32_t shift_bits = (shift & 0x3F) << 10;

    // insert the new bits
    l_ins |= shift_bits | wn_bits | wd_bits;

    // determin w or x view:
    uint32_t l_reg_size = Wd & 0x20;
    l_ins |= l_reg_size << 26 | l_reg_size << 22;

    return l_ins;
}

uint32_t jiter::instructions::InstGen::base_lsl_register(gpr_t Wd, gpr_t Wn, gpr_t Wm) {
    uint32_t l_ins = 0x1AC02000;

    // extract lower 5 bits from each and shift them to correct position
    uint32_t wd_bits = (Wd & 0x1F) << 0;
    uint32_t wn_bits = (Wn & 0x1F) << 5;
    uint32_t wm_bits = (Wm & 0x1F) << 16;

    // insert the new bits
    l_ins |= wm_bits | wn_bits | wd_bits;

    // determin w or x view:
    uint32_t l_reg_size = Wd & 0x20;
    l_ins |= l_reg_size << 26 | l_reg_size << 22;

    return l_ins;
}

uint32_t jiter::instructions::InstGen::base_ret(gpr_t Xn) {
    uint32_t l_ins = 0xD65F0000;

    // extract lower bits from each and shift them to correct position
    uint32_t xn_bits = (Xn & 0x1F) << 5;

    // insert the new bits
    l_ins |= xn_bits;

    return l_ins;
}
