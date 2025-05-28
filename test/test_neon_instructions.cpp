#include "../src/mini_jit/instructions/instructions.h"
#include "./test_utils.h"

int test_neon_fmla_element() {
    using namespace mini_jit::instructions;

    uint32_t mc1 = InstGen::neon_fmla_element(InstGen::simd_fp_t::v8,
                                              InstGen::simd_fp_t::v8,
                                              InstGen::simd_fp_t::v9,
                                              InstGen::element_spec_t::D2_1);
    std::string call = "fmla v8.2d, v8.2d, v9.d[1]";
    uint32_t mc2 = test_utils::as(call);

    return test_utils::instr_is_correct(call, mc1, mc2);
}

int test_neon_movi_zero() {
    using namespace mini_jit::instructions;

    uint32_t mc1 = InstGen::neon_movi_zero(InstGen::simd_fp_t::v8,
                                           true,
                                           false);
    std::string call = "movi v8.4s, #0";
    uint32_t mc2 = test_utils::as(call);

    return test_utils::instr_is_correct(call, mc1, mc2);
}

int test_neon_fmaxnmp_vector() {
    using namespace mini_jit::instructions;

    uint32_t mc1 = InstGen::neon_fmaxnmp_vector(InstGen::simd_fp_t::v8,
                                                InstGen::simd_fp_t::v9,
                                                InstGen::simd_fp_t::v7,
                                                false);
    std::string call = "fmaxnmp v8.4s, v9.4s, v7.4s";
    uint32_t mc2 = test_utils::as(call);

    return test_utils::instr_is_correct(call, mc1, mc2);
}

int test_neon_fmax_vector() {
    using namespace mini_jit::instructions;

    uint32_t mc1 = InstGen::neon_fmax_vector(InstGen::simd_fp_t::v0,
                                             InstGen::simd_fp_t::v1,
                                             InstGen::simd_fp_t::v2,
                                             true);
    std::string call = "fmax v0.2d, v1.2d, v2.2d";
    uint32_t mc2 = test_utils::as(call);

    return test_utils::instr_is_correct(call, mc1, mc2);
}

int test_neon_ld1() {
    using namespace mini_jit::instructions;

    uint32_t mc1 = InstGen::neon_ld1_multiple(InstGen::simd_fp_t::v6,
                                              InstGen::gpr_t::x0,
                                              InstGen::ld1_opcode_t::three_regs,
                                              InstGen::ld1_t::S4);
    std::string call = "ld1 {v6.4s - v8.4s}, [x0]";
    uint32_t mc2 = test_utils::as(call);

    return test_utils::instr_is_correct(call, mc1, mc2);
}

int test_neon_st1() {
    using namespace mini_jit::instructions;

    uint32_t mc1 = InstGen::neon_st1_multiple(InstGen::simd_fp_t::v10,
                                              InstGen::gpr_t::x30,
                                              InstGen::ld1_opcode_t::two_regs,
                                              InstGen::ld1_t::S4);
    std::string call = "st1 {v10.4s, v11.4s}, [x30]";
    uint32_t mc2 = test_utils::as(call);

    return test_utils::instr_is_correct(call, mc1, mc2);
}

int main() {
    int result = 0;

    result |= test_neon_fmla_element();
    result |= test_neon_movi_zero();
    result |= test_neon_fmaxnmp_vector();
    result |= test_neon_fmax_vector();
    result |= test_neon_ld1();
    result |= test_neon_st1();

    return result;
}
