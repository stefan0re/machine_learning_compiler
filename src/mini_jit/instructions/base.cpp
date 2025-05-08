#include "instructions.h"


namespace mini_jit {
    namespace instructions {

        // cbnz  <W/X><Rt>, #+imm19
        uint32_t InstGen::base_br_cbnz(gpr_t Rt, uint32_t imm19) {
            uint32_t ins = 0x35000000u;
            ins |= (Rt & 0x1Fu);                // Rt → bits [4:0]
            ins |= (((Rt >> 5) & 0x1u) << 31);  // sf → bit 31
            ins |= (imm19 & 0x7FFFFu) << 5;     // imm19 → bits [23:5]
            return ins;
        }

        // ldp  <W/X>t1, <W/X>t2, [<Xn|SP>], #+imm7
        uint32_t InstGen::base_ldp(gpr_t t1, gpr_t t2, gpr_t Xn_SP, uint32_t imm7) {
            uint32_t ins = 0x28C00000u;
            ins |= (t1 & 0x1Fu) << 0;           // Rt1 → [4:0]
            ins |= (Xn_SP & 0x1Fu) << 5;        // Rn  → [9:5]
            ins |= (t2 & 0x1Fu) << 10;          // Rt2 → [14:10]
            ins |= ((imm7 & 0x7Fu) << 15);      // imm7→ [21:15]
            ins |= (((t1 >> 5) & 0x1u) << 31);  // sf → bit 31
            return ins;
        }

        // stp  <W/X>t1, <W/X>t2, [<Xn|SP>], #+imm7
        uint32_t InstGen::base_stp(gpr_t t1, gpr_t t2, gpr_t Xn_SP, uint32_t imm7) {
            uint32_t ins = 0x28800000u;
            ins |= (t1 & 0x1Fu) << 0;
            ins |= (Xn_SP & 0x1Fu) << 5;
            ins |= (t2 & 0x1Fu) << 10;
            ins |= ((imm7 & 0x7Fu) << 15);
            ins |= (((t1 >> 5) & 0x1u) << 31);
            return ins;
        }

        // mov  <W/X>d, #imm12   (alias of ORR Wd, WZR, #imm12)
        uint32_t InstGen::base_mov_imm(gpr_t Wd, uint16_t imm16, uint8_t shift /*= 0*/) {
            uint32_t ins = 0x52800000u;       // MOVZ base for 32-bit (sf = 0)
            ins |= ((imm16 & 0xFFFFu) << 5);  // imm16 → bits [20:5]
            ins |= ((shift & 0x3u) << 21);    // shift amount (0, 16, 32, 48) → bits [22:21]
            ins |= (Wd & 0x1Fu);              // Rd → bits [4:0]
            ins |= ((Wd >> 5) & 0x1u) << 31;  // sf (bit 31): 0 = 32-bit, 1 = 64-bit
            return ins;
        }

        // mov  <W/X>d, <W/X>m
        uint32_t InstGen::base_mov_register(gpr_t Wd, gpr_t Wm) {
            uint32_t ins = 0x2A0003E0u;  // ORR (reg) base
            ins |= (((Wd >> 5) & 0x1u) << 31);
            ins |= ((Wm & 0x1Fu) << 16);  // Rm → bits [20:16]
            ins |= (Wd & 0x1Fu);          // Rd → bits [4:0]
            return ins;
        }

        // add  <W/X>d, <W/X>n, #imm12 {, LSL #shift}
        uint32_t InstGen::base_add_imm(gpr_t Wd, gpr_t Wn, uint32_t imm12, uint32_t shift) {
            uint32_t ins = 0x11000000u;
            ins |= (((Wd >> 5) & 0x1u) << 31);
            ins |= ((shift & 1u) << 22);  // LSL #shift? only 0 or 1
            ins |= (imm12 & 0xFFFu) << 10;
            ins |= ((Wn & 0x1Fu) << 5);
            ins |= (Wd & 0x1Fu);
            return ins;
        }

        // add  <W/X>d, <W/X>n, <W/X>m, {LSL|LSR|ASR} #imm6
        uint32_t InstGen::base_add_shifted_register(
            gpr_t Wd, gpr_t Wn, gpr_t Wm, uint32_t shift_type, uint32_t imm6) {
            uint32_t ins = 0x0B000000u;
            ins |= (((Wd >> 5) & 0x1u) << 31);
            ins |= ((shift_type & 0x3u) << 22);
            ins |= ((imm6 & 0x3Fu) << 10);
            ins |= ((Wm & 0x1Fu) << 16);
            ins |= ((Wn & 0x1Fu) << 5);
            ins |= (Wd & 0x1Fu);
            return ins;
        }

        // sub  <W/X>d, <W/X>n, #imm12 {, LSL #shift}
        uint32_t InstGen::base_sub_imm(gpr_t Wd, gpr_t Wn, uint32_t imm12, uint32_t shift) {
            uint32_t ins = 0x51000000u;
            ins |= (((Wd >> 5) & 0x1u) << 31);
            ins |= ((shift & 1u) << 22);
            ins |= (imm12 & 0xFFFu) << 10;
            ins |= ((Wn & 0x1Fu) << 5);
            ins |= (Wd & 0x1Fu);
            return ins;
        }

        // sub  <W/X>d, <W/X>n, <W/X>m, {LSL|LSR|ASR} #imm6
        uint32_t InstGen::base_sub_shifted_register(
            gpr_t Wd, gpr_t Wn, gpr_t Wm, uint32_t shift_type, uint32_t imm6) {
            uint32_t ins = 0x4B000000u;
            ins |= (((Wd >> 5) & 0x1u) << 31);
            ins |= ((shift_type & 0x3u) << 22);
            ins |= ((imm6 & 0x3Fu) << 10);
            ins |= ((Wm & 0x1Fu) << 16);
            ins |= ((Wn & 0x1Fu) << 5);
            ins |= (Wd & 0x1Fu);
            return ins;
        }

        // lsl  <W/X>d, <W/X>n, #imm6
        uint32_t InstGen::base_lsl_imm(gpr_t Wd, gpr_t Wn, uint32_t shift) {
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

        // lsl  <W/X>d, <W/X>n, <W/X>m
        uint32_t InstGen::base_lsl_register(gpr_t Wd, gpr_t Wn, gpr_t Wm) {
            uint32_t ins = 0x1AC02000u;
            ins |= (((Wd >> 5) & 0x1u) << 31);
            ins |= ((Wm & 0x1Fu) << 16);
            ins |= ((Wn & 0x1Fu) << 5);
            ins |= (Wd & 0x1Fu);
            return ins;
        }

        uint32_t InstGen::base_ret() {
            return 0xd65f03c0;
        }

        uint32_t InstGen::base_mul_reg(gpr_t dst,
                                       gpr_t src_1,
                                       gpr_t src_0) {
            uint32_t l_ins = 0x1b007c00;

            l_ins |= (0x1f & dst);
            l_ins |= (0x1f & src_1) << 5;
            l_ins |= (0x1f & src_0) << 16;
            l_ins |= (0x20 & dst) << 26;

            return l_ins;
        }


    }  // namespace instructions
}  // namespace mini_jit
