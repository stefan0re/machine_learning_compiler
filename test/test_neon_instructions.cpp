#include <iostream>

#include "../src/mini_jit/instructions/instructions.h"

int test_neon_fmla_element() {
    using namespace mini_jit::instructions;

    std::cout << "Testing Neon FMLA (by element) instruction generation ... ";
    uint32_t instruction = InstGen::neon_fmla_element(InstGen::simd_fp_t::v8,
                                                      InstGen::simd_fp_t::v8,
                                                      InstGen::simd_fp_t::v9,
                                                      InstGen::element_spec_t::s_1);
    uint32_t expected = 0x5FD91108;

    if (instruction != expected) {
        std::cout << "Failed" << std::endl;
        return -1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}

int main() {
    int result = 0;

    result |= test_neon_fmla_element();

    return result;
}
