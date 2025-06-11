#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../../src/mini_jit/instructions/instructions.h"

using namespace mini_jit::instructions;

uint32_t as(const std::string& instruction) {
    // write the instruction to a temporary assembly file
    std::ofstream asmFile("temp.s");
    asmFile << ".text\n.global _start\n_start:\n    " << instruction << "\n";
    asmFile.close();

    // assemble it to an object file
    if (system("as temp.s -o temp.o") != 0) {
        throw std::runtime_error("Assembly failed");
    }

    // extract raw binary
    if (system("objcopy -O binary temp.o temp.bin") != 0) {
        throw std::runtime_error("Objcopy failed");
    }

    // read first 4 bytes of binary output
    std::ifstream binFile("temp.bin", std::ios::binary);
    std::array<char, 4> bytes{};
    binFile.read(bytes.data(), 4);
    binFile.close();

    // convert to uint32_t
    uint32_t result = static_cast<unsigned char>(bytes[0]) |
                      (static_cast<unsigned char>(bytes[1]) << 8) |
                      (static_cast<unsigned char>(bytes[2]) << 16) |
                      (static_cast<unsigned char>(bytes[3]) << 24);
    return result;
}

TEST_CASE("MiniJit::Instructions::Encoding::base_ldp", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_ldp(InstGen::gpr_t::w1,
                                     InstGen::gpr_t::w2,
                                     InstGen::gpr_t::x3, 0);
    std::string call = "ldp w1, w2, [x3], #0";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_stp", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_stp(InstGen::gpr_t::w1,
                                     InstGen::gpr_t::w2,
                                     InstGen::gpr_t::x3, 0);
    std::string call = "stp w1, w2, [x3], #0";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_mov_imm", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_mov_imm(InstGen::gpr_t::w1, 1, 16);
    std::string call = "mov w1, #1";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_mov_register", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_mov_register(InstGen::gpr_t::w1, InstGen::gpr_t::w2);
    std::string call = "mov w1, w2";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_movz", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_movz(InstGen::gpr_t::x1, 3, 48);
    std::string call = "movz x1, #3, lsl #48";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_movk", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_movk(InstGen::gpr_t::x7, 22, 0);
    std::string call = "movk x7, #22, lsl #0";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_add_imm", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_add_imm(InstGen::gpr_t::w1, InstGen::gpr_t::w2, 1, 0);
    std::string call = "add w1, w2, #1";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_add_shifted_register", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_add_shifted_register(InstGen::gpr_t::w1, InstGen::gpr_t::w2, InstGen::gpr_t::w3, 0, 0);
    std::string call = "add w1, w2, w3";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_sub_imm", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_sub_imm(InstGen::gpr_t::w1, InstGen::gpr_t::w2, 1, 0);
    std::string call = "sub w1, w2, #1";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_sub_shifted_register", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_sub_shifted_register(InstGen::gpr_t::w1, InstGen::gpr_t::w2, InstGen::gpr_t::w3, 0, 0);
    std::string call = "sub w1, w2, w3";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_lsl_imm", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_lsl_imm(InstGen::gpr_t::w1, InstGen::gpr_t::w2, 0);
    std::string call = "lsl w1, w2, #0";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_lsl_register", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_lsl_register(InstGen::gpr_t::w1, InstGen::gpr_t::w2, InstGen::gpr_t::w3);
    std::string call = "lsl w1, w2, w3";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_br_cbnz", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = InstGen::base_br_cbnz(InstGen::gpr_t::w1, 1);
    std::string call = "cbnz w1, 0x00000004";
    uint32_t mc2 = as(call);
    REQUIRE(mc1 == mc2);
}
