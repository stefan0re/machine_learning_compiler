#include "./test_utils.h"

std::bitset<32> test_utils::get_binary(uint32_t decimal) {
    std::bitset<32> binary(decimal);
    return binary;
}

uint32_t test_utils::as(const std::string& instruction) {
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

int test_utils::instr_is_correct(std::string call, uint32_t result, uint32_t expected) {
    bool match = (result == expected);
    std::cout << call << ": " << std::boolalpha << match << "\n"
              << "result:  " << test_utils::get_binary(result) << "\n"
              << "correct: " << test_utils::get_binary(expected)
              << std::endl;
    return match ? 0 : -1;
}

void test_utils::generate_matrix(uint32_t height, uint32_t width, float* M, bool set_zero) {
    float MAX = 100;
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int index = j * height + i;
            float element = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / MAX));
            M[index] = (1 - (double)set_zero) * element;
        }
    }
}

void test_utils::visualize_matrix(uint32_t height, uint32_t width, float* M, std::string name) {
    std::cout << name << std::endl;
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int index = j * height + i;
            std::cout << M[index] << " ";
        }
        std::cout << std::endl;
    }
}

bool test_utils::compare_matrix(uint32_t height, uint32_t width, float* M, float* C) {
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int index = j * height + i;
            if (M[index] != C[index]) {
                std::cout << "Matrices are not equal in element: " << index << " M: " << M[index] << ", C: " << C[index] << std::endl;
                return false;
            }
        }
    }
    return true;
}