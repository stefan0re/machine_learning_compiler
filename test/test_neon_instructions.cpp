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

int main() {
    int result = 0;

    result |= test_neon_fmla_element();
    result |= test_neon_movi_zero();

    return result;
}
