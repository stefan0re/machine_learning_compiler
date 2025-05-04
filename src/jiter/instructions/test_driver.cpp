#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "instructions.h"

using namespace std;
using gpr_t = jiter::instructions::InstGen::gpr_t;

uint32_t as(const string& instruction) {
    // write the instruction to a temporary assembly file
    ofstream asmFile("temp.s");
    asmFile << ".text\n.global _start\n_start:\n    " << instruction << "\n";
    asmFile.close();

    // assemble it to an object file
    system("as temp.s -o temp.o");

    // extract raw binary
    system("objcopy -O binary temp.o temp.bin");

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

int main(int argc, char const* argv[]) {
    jiter::instructions::InstGen l_gen;

    cout << "Tests:" << endl;

    bool match = l_gen.base_br_cbnz(gpr_t::w1, 10) == as("cbnz w1, #10");
    cout << "cbnz w1, #10: " << std::boolalpha << match << endl;

    return 0;
}
