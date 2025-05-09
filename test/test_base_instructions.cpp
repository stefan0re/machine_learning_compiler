#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../src/mini_jit/instructions/instructions.h"

using namespace std;
using gpr_t = mini_jit::instructions::InstGen::gpr_t;
mini_jit::instructions::InstGen l_gen;

uint32_t as(const string& instruction) {
    // write the instruction to a temporary assembly file
    ofstream asmFile("temp.s");
    asmFile << ".text\n.global _start\n_start:\n    " << instruction << "\n";
    asmFile.close();

    // assemble it to an object file
    if (system("as temp.s -o temp.o") != 0) {
        throw runtime_error("Assembly failed");
    }

    // extract raw binary
    if (system("objcopy -O binary temp.o temp.bin") != 0) {
        throw runtime_error("Objcopy failed");
    }

    // read first 4 bytes of binary output
    ifstream binFile("temp.bin", ios::binary);
    array<char, 4> bytes{};
    binFile.read(bytes.data(), 4);
    binFile.close();

    // convert to uint32_t
    uint32_t result = static_cast<unsigned char>(bytes[0]) |
                      (static_cast<unsigned char>(bytes[1]) << 8) |
                      (static_cast<unsigned char>(bytes[2]) << 16) |
                      (static_cast<unsigned char>(bytes[3]) << 24);
    return result;
}

int test_base_br_cbnz() {
    uint32_t mc1 = l_gen.base_br_cbnz(gpr_t::w1, 1);
    uint32_t mc2 = as("cbnz w1, 0x00000004");
    bool match = (mc1 == mc2);
    cout << "cbnz w1, 0x00000001: "
         << mc1 << " | " << mc2 << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
}

int test_base_ldp() {
    uint32_t mc1 = l_gen.base_ldp(gpr_t::w1, gpr_t::w2, gpr_t::x3, 0);
    uint32_t mc2 = as("ldp w1, w2, [x3], #0");
    bool match = (mc1 == mc2);
    cout << "ldp w1, w2, [x3], #0: "
         << mc1 << " | " << mc2 << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
}

int test_base_stp() {
    uint32_t mc1 = l_gen.base_stp(gpr_t::w1, gpr_t::w2, gpr_t::x3, 0);
    uint32_t mc2 = as("stp w1, w2, [x3], #0");
    bool match = (mc1 == mc2);
    cout << "stp w1, w2, [x3], #0: "
         << mc1 << " | " << mc2 << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
}

int test_base_mov_imm() {
    uint32_t mc1 = l_gen.base_mov_imm(gpr_t::w1, 1, 0);
    uint32_t mc2 = as("mov w1, #1");
    bool match = (mc1 == mc2);
    cout << "mov w1, #1: "
         << mc1 << " | " << mc2 << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
}

int test_base_mov_register() {
    uint32_t mc1 = l_gen.base_mov_register(gpr_t::w1, gpr_t::w2);
    uint32_t mc2 = as("mov w1, w2");
    bool match = (mc1 == mc2);
    cout << "mov w1, w2: "
         << mc1 << " | " << mc2 << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
}

int test_base_add_imm() {
    uint32_t mc1 = l_gen.base_add_imm(gpr_t::w1, gpr_t::w2, 1, 0);
    uint32_t mc2 = as("add w1, w2, #1");
    bool match = (mc1 == mc2);
    cout << "add w1, w2, #1: "
         << mc1 << " | " << mc2 << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
}

int test_base_add_shifted_register() {
    uint32_t mc1 = l_gen.base_add_shifted_register(gpr_t::w1, gpr_t::w2, gpr_t::w3, 0, 0);
    uint32_t mc2 = as("add w1, w2, w3");
    bool match = (mc1 == mc2);
    cout << "add w1, w2, w3: "
         << mc1 << " | " << mc2 << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
}

int test_base_sub_imm() {
    uint32_t mc1 = l_gen.base_sub_imm(gpr_t::w1, gpr_t::w2, 1, 0);
    uint32_t mc2 = as("sub w1, w2, #1");
    bool match = (mc1 == mc2);
    cout << "sub w1, w2, #1: "
         << mc1 << " | " << mc2 << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
}

int test_base_sub_shifted_register() {
    uint32_t mc1 = l_gen.base_sub_shifted_register(gpr_t::w1, gpr_t::w2, gpr_t::w3, 0, 0);
    uint32_t mc2 = as("sub w1, w2, w3");
    bool match = (mc1 == mc2);
    cout << "sub w1, w2, w3: "
         << mc1 << " | " << mc2 << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
}

int test_base_lsl_imm() {
    uint32_t mc1 = l_gen.base_lsl_imm(gpr_t::w1, gpr_t::w2, 0);
    uint32_t mc2 = as("lsl w1, w2, #0");
    bool match = (mc1 == mc2);
    cout << "lsl w1, w2, #0: "
         << mc1 << " | " << mc2 << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
}

int test_base_lsl_register() {
    uint32_t mc1 = l_gen.base_lsl_register(gpr_t::w1, gpr_t::w2, gpr_t::w3);
    uint32_t mc2 = as("lsl w1, w2, w3");
    bool match = (mc1 == mc2);
    cout << "lsl w1, w2, w3: "
         << mc1 << " | " << mc2 << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
}

int test_base_br_cbnz() {
    uint32_t instr = l_gen.base_br_cbnz(gpr_t::w1, 1);
    uint32_t expected = as("cbnz w1, 0x00000004");
    bool match = (instr == expected);

    cout << "cbnz w1, 0x00000004: "
         << instr << " | " << expected << " : "
         << boolalpha << match << endl;
    return match ? 0 : -1;
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
