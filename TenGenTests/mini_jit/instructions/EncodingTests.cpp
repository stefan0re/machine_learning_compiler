#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "TenGenTestsHelper.h"

using namespace TenGen::Types;
using namespace TenGen::MiniJit::Instructions::Encoding;
using namespace std;

TEST_CASE("MiniJit::Instructions::Encoding::base_ldp", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_ldp(gpr_t::w1, gpr_t::w2, gpr_t::x3, 0);
    std::string call = "ldp w1, w2, [x3], #0";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_stp", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_stp(gpr_t::w1, gpr_t::w2, gpr_t::x3, 0);
    std::string call = "stp w1, w2, [x3], #0";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_mov_imm", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_mov_imm(gpr_t::w1, 1, 16);
    std::string call = "mov w1, #1";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_mov_register", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_mov_register(gpr_t::w1, gpr_t::w2);
    std::string call = "mov w1, w2";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_movz", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_movz(gpr_t::x1, 3, 48);
    std::string call = "movz x1, #3, lsl #48";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_movk", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_movk(gpr_t::x7, 22, 0);
    std::string call = "movk x7, #22, lsl #0";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_add_imm", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_add_imm(gpr_t::w1, gpr_t::w2, 1, 0);
    std::string call = "add w1, w2, #1";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_add_shifted_register", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_add_shifted_register(gpr_t::w1, gpr_t::w2, gpr_t::w3, 0, 0);
    std::string call = "add w1, w2, w3";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_sub_imm", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_sub_imm(gpr_t::w1, gpr_t::w2, 1, 0);
    std::string call = "sub w1, w2, #1";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_sub_shifted_register", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_sub_shifted_register(gpr_t::w1, gpr_t::w2, gpr_t::w3, 0, 0);
    std::string call = "sub w1, w2, w3";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_lsl_imm", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_lsl_imm(gpr_t::w1, gpr_t::w2, 0);
    std::string call = "lsl w1, w2, #0";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_lsl_register", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_lsl_register(gpr_t::w1, gpr_t::w2, gpr_t::w3);
    std::string call = "lsl w1, w2, w3";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}

TEST_CASE("MiniJit::Instructions::Encoding::base_br_cbnz", "[MiniJit][Instructions][Encoding]") {
    uint32_t mc1 = base_br_cbnz(gpr_t::w1, 1);
    std::string call = "cbnz w1, 0x00000004";
    uint32_t mc2 = TenGenTestsHelper::as(call);
    REQUIRE(mc1 == mc2);
}