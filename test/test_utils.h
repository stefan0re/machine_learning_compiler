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
    std::bitset<32> get_binary(std::uint32_t decimal);
}  // namespace test_utils

#endif