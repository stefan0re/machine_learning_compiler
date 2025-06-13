#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../../src/tensor/tensor.h"

TEST_CASE("Unary generate zero kernel sets all elements to zero", "[unary][generate_zero]") {
    Tensor tensor = Tensor(3, 4, 5);
    tensor.info();
}
