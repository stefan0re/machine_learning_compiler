#include <iostream>

#include "../src/mini_jit/instructions/instructions.h"

int test_base_br_cbnz() {
    using namespace mini_jit::instructions;

    std::cout << "Testing Branching CBNZ instruction generation ... ";
    uint32_t instruction = InstGen::base_br_cbnz(InstGen::w0, 0x12345);
    uint32_t expected = 0x35000000 | (0x0 & 0x1f) | ((0x0 & 0x20) << (32 - 6)) | (0x12345 << 5);

    if (instruction != expected) {
        std::cout << "Failed" << std::endl;
        return -1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}

int main() {
    int result = 0;

    result |= test_base_br_cbnz();

    return result;
}
