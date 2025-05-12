#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <array>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
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
}  // namespace test_utils

#endif