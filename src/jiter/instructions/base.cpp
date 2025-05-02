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