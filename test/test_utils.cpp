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