#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <array>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

namespace test_utils {
    /**
     * @brief Generates machine code for an assembly instruction in form of a string.
     *
     * @param instruction assembly instruction in form of a string.
     *
     * @return machine code encoding.
     **/
    uint32_t as(const std::string& instruction);

    /**
     * @brief Generates binary representation of a decimal number.
     *
     * @param decimal decimal number.
     *
     * @return binary representation.
     **/
    std::bitset<32> get_binary(uint32_t decimal);

    /**
     * @brief Checks if result and expected are equal, and prints and returns the result for unit testing.
     *
     * @param call respective assembly call.
     * @param result number to be tested.
     * @param expected correct encoding of the assembly call.
     *
     * @return if result==expected 0, else 1.
     **/
    int instr_is_correct(std::string call, uint32_t result, uint32_t expected);

    /**
     @brief Generate a matrix with random numbers (or zeros) as elements.
     *
     * @param height m dimension of the produced matrix.
     * @param width n dimension of the produced matrix.
     * @param M pointer to the produced matrix.
     * @param set_zero if true, all elements of the produced matrix are set zero.
    */
    void generate_matrix(uint32_t height, uint32_t width, float* M, bool set_zero = false);

    /**
     @brief Visualize matrix in terminal.
     *
     * @param height m dimension of the produced matrix.
     * @param width n dimension of the produced matrix.
     * @param M pointer to the produced matrix.
     * @param name contextual name of the matrix.
    */
    void visualize_matrix(uint32_t height, uint32_t width, float* M, std::string name);

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
    bool compare_matrix(uint32_t height, uint32_t width, float* M, float* C);
}  // namespace test_utils

#endif