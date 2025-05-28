#ifndef TenGenTestsHelper_H
#define TenGenTestsHelper_H

#include <array>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

namespace TenGenTestsHelper {
    /**
     * @brief Generates machine code for an assembly instruction in form of a string.
     *
     * @param instruction assembly instruction in form of a string.
     *
     * @return machine code encoding.
     **/
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

    /**
     * @brief Generates binary representation of a decimal number.
     *
     * @param decimal decimal number.
     *
     * @return binary representation.
     **/
    std::bitset<32> get_binary(uint32_t decimal) {
        std::bitset<32> binary(decimal);
        return binary;
    }

    /**
     * @brief Checks if result and expected are equal, and prints and returns the result for unit testing.
     *
     * @param call respective assembly call.
     * @param result number to be tested.
     * @param expected correct encoding of the assembly call.
     *
     * @return if result==expected 0, else 1.
     **/
    int instr_is_correct(std::string call, uint32_t result, uint32_t expected) {
        bool match = (result == expected);
        std::cout << call << ": " << std::boolalpha << match << "\n"
                  << "result:  " << TenGenTestsHelper::get_binary(result) << "\n"
                  << "correct: " << TenGenTestsHelper::get_binary(expected)
                  << std::endl;
        return match ? 0 : -1;
    }

    /**
     @brief Generate a matrix with random numbers (or zeros) as elements.
     *
     * @param height m dimension of the produced matrix.
     * @param width n dimension of the produced matrix.
     * @param M pointer to the produced matrix.
     * @param set_zero if true, all elements of the produced matrix are set zero.
    */
    void generate_matrix(uint32_t height, uint32_t width, float* M, bool set_zero = false, bool visualization = false) {
        float MAX = 100;
        float MIN = -100;
        for (uint32_t i = 0; i < height; i++) {
            for (uint32_t j = 0; j < width; j++) {
                int index = j * height + i;
                float rand_float = static_cast<float>(rand()) / RAND_MAX;  // [0,1]
                rand_float = rand_float * (MAX - MIN) + MIN;
                if (visualization) {
                    rand_float = std::trunc(rand_float);
                }
                M[index] = (1 - (double)set_zero) * rand_float;
            }
        }
    }

    /**
     @brief Transpose (Identity) a given matrix.
    *
    * @param height m dimension of the matrix.
    * @param width n dimension of the matrix.
    * @param M pointer to the matrix that should transposed.
    * @param N pointer to the matrix that holds the final transposed matrix.
    */
    void transpose_matrix(uint32_t height, uint32_t width, float* M, float* N) {
        for (uint32_t i = 0; i < height; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                N[j * height + i] = M[i * width + j];
            }
        }
    }

    /**
     @brief Visualize matrix in terminal.
     *
     * @param height m dimension of the produced matrix.
     * @param width n dimension of the produced matrix.
     * @param M pointer to the produced matrix.
     * @param name contextual name of the matrix.
    */
    void visualize_matrix(uint32_t height, uint32_t width, float* M, std::string name) {
        std::cout << name << std::endl;
        for (uint32_t i = 0; i < height; i++) {
            for (uint32_t j = 0; j < width; j++) {
                int index = j * height + i;
                std::cout << M[index] << " ";
            }
            std::cout << std::endl;
        }
    }

    /**
     @brief Checks if result and expected are equal, and prints and returns the result for unit testing.
     *
     * @param height m dimension of the matrices.
     * @param width n dimension of the matrices.
     * @param M pointer to the first matrix.
     * @param C pointer to the second matrix.
     *
     * @return if worke correct 1, else 0.
    */
    bool compare_matrix(uint32_t height, uint32_t width, float* M, float* C) {
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
}  // namespace TenGenTestsHelper

#endif  // TenGenTestsHelper_H