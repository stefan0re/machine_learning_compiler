#include <catch2/catch_test_macros.hpp>

#include "TenGen.h"

TEST_CASE("Copy Tensors", "[Tensor][Tensor]") {
    // create a TenGen tensor and fill it with random values
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor1(true);
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor2(true);

    std::copy_n(tenGenTensor1.exportPointer(), tenGenTensor2.size, tenGenTensor2.exportPointer());

    REQUIRE(tenGenTensor1 == tenGenTensor2);
}
