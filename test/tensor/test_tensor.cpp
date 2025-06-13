#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../../src/tensor/tensor.cpp"
#include "../../src/tensor/tensor.h"

TEST_CASE("Tensor.info()", "[tensor][info]") {
    Tensor tensor = Tensor(3, 4, 5);

    INFO(tensor.info_str());
    REQUIRE(false);
}

TEST_CASE("Tensor.swap()", "[tensor][info]") {
    Tensor tensor = Tensor(3, 4, 5);

    INFO(tensor.info_str());

    tensor.swap(1, 2);

    INFO(tensor.info_str());

    REQUIRE(false);
}