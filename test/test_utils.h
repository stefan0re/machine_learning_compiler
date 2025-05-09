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
    uint32_t as(const std::string& instruction);
    std::bitset<32> get_binary(uint32_t decimal);
    int is_correct(std::string call, uint32_t result, uint32_t expected);
}  // namespace test_utils

#endif