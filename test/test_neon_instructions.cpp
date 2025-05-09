#include "../src/mini_jit/instructions/instructions.h"
#include "./test_utils.h"

int test_neon_fmla_element() {
    using namespace mini_jit::instructions;
    uint32_t mc1 = InstGen::neon_fmla_element(InstGen::simd_fp_t::v8,
                                              InstGen::simd_fp_t::v8,
                                              InstGen::simd_fp_t::v9,
                                              InstGen::element_spec_t::S4_1);
    uint32_t mc2 = test_utils::as("fmla v8.4s, v8.4s, v9.s[1]");

    bool match = (mc1 == mc2);
    std::cout << "stp w1, w2, [x3], #0: "
              << test_utils::get_binary(mc1) << " | " << test_utils::get_binary(mc2) << " : "
              << std::boolalpha << match << std::endl;
    return match ? 0 : -1;
}

int main() {
    int result = 0;

    result |= test_neon_fmla_element();

    return result;
}
