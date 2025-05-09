#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../src/mini_jit/instructions/instructions.h"
#include "./test_utils.h"

using namespace std;
using gpr_t = mini_jit::instructions::InstGen::gpr_t;
mini_jit::instructions::InstGen l_gen;

int test_base_ldp() {
    uint32_t mc1 = l_gen.base_ldp(gpr_t::w1, gpr_t::w2, gpr_t::x3, 0);
    string call = "ldp w1, w2, [x3], #0";
    uint32_t mc2 = test_utils::as(call);
    return test_utils::is_correct(call, mc1, mc2);
}

int test_base_stp() {
    uint32_t mc1 = l_gen.base_stp(gpr_t::w1, gpr_t::w2, gpr_t::x3, 0);
    string call = "stp w1, w2, [x3], #0";
    uint32_t mc2 = test_utils::as(call);
    return test_utils::is_correct(call, mc1, mc2);
}

int test_base_mov_imm() {
    uint32_t mc1 = l_gen.base_mov_imm(gpr_t::w1, 1, 0);
    string call = "mov w1, #1";
    uint32_t mc2 = test_utils::as(call);
    return test_utils::is_correct(call, mc1, mc2);
}

int test_base_mov_register() {
    uint32_t mc1 = l_gen.base_mov_register(gpr_t::w1, gpr_t::w2);
    string call = "mov w1, w2";
    uint32_t mc2 = test_utils::as(call);
    return test_utils::is_correct(call, mc1, mc2);
}

int test_base_add_imm() {
    uint32_t mc1 = l_gen.base_add_imm(gpr_t::w1, gpr_t::w2, 1, 0);
    string call = "add w1, w2, #1";
    uint32_t mc2 = test_utils::as(call);
    return test_utils::is_correct(call, mc1, mc2);
}

int test_base_add_shifted_register() {
    uint32_t mc1 = l_gen.base_add_shifted_register(gpr_t::w1, gpr_t::w2, gpr_t::w3, 0, 0);
    string call = "add w1, w2, w3";
    uint32_t mc2 = test_utils::as(call);
    return test_utils::is_correct(call, mc1, mc2);
}

int test_base_sub_imm() {
    uint32_t mc1 = l_gen.base_sub_imm(gpr_t::w1, gpr_t::w2, 1, 0);
    string call = "sub w1, w2, #1";
    uint32_t mc2 = test_utils::as(call);
    return test_utils::is_correct(call, mc1, mc2);
}

int test_base_sub_shifted_register() {
    uint32_t mc1 = l_gen.base_sub_shifted_register(gpr_t::w1, gpr_t::w2, gpr_t::w3, 0, 0);
    string call = "sub w1, w2, w3";
    uint32_t mc2 = test_utils::as(call);
    return test_utils::is_correct(call, mc1, mc2);
}

int test_base_lsl_imm() {
    uint32_t mc1 = l_gen.base_lsl_imm(gpr_t::w1, gpr_t::w2, 0);
    string call = "lsl w1, w2, #0";
    uint32_t mc2 = test_utils::as(call);
    return test_utils::is_correct(call, mc1, mc2);
}

int test_base_lsl_register() {
    uint32_t mc1 = l_gen.base_lsl_register(gpr_t::w1, gpr_t::w2, gpr_t::w3);
    string call = "lsl w1, w2, w3";
    uint32_t mc2 = test_utils::as(call);
    return test_utils::is_correct(call, mc1, mc2);
}

int test_base_br_cbnz() {
    uint32_t mc1 = l_gen.base_br_cbnz(gpr_t::w1, 1);
    string call = "cbnz w1, 0x00000004";
    uint32_t mc2 = test_utils::as(call);
    return test_utils::is_correct(call, mc1, mc2);
}

int main() {
    int result = 0;

    result |= test_base_br_cbnz();
    result |= test_base_ldp();
    result |= test_base_stp();
    result |= test_base_mov_imm();
    result |= test_base_mov_register();
    result |= test_base_add_imm();
    result |= test_base_add_shifted_register();
    result |= test_base_sub_imm();
    result |= test_base_sub_shifted_register();
    result |= test_base_lsl_imm();
    result |= test_base_lsl_register();

    return result;
}
