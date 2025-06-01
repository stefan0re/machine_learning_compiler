#include <catch2/catch_test_macros.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

#include "TenGen.h"
#include "TenGenTests/TenGenTestsHelper.h"

using namespace Fastor;

TEST_CASE("Fastor Tensor == TenGen Tensor", "[Tensor][Tensor]") {
    // create a TenGen tensor and fill it with random values
    TenGen::Tensor<float, 2, 2> tenGenTensor1;
    tenGenTensor1.fillRandom();
    TenGen::Tensor<float, 2, 2> tenGenTensor2;
    tenGenTensor2.fillRandom();

    // create a xtensor (real copy) from the TenGen tensor
    auto xt_tensor1 = xt::adapt(tenGenTensor1.exportPointer(), tenGenTensor1.size(), xt::ownership(), {2, 2});
    auto xt_tensor2 = xt::adapt(tenGenTensor2.exportPointer(), tenGenTensor2.size(), xt::ownership(), {2, 2});

    // do einstein summation on both tensors with xtensor
    auto xt_tensor3 = xt::linalg::einsum("mk,kn->mn", xt_tensor1, xt_tensor2);

    TenGen::Tensor<float, 2, 2> tenGenTensor3;
    tenGenTensor3.fromXTensor(xt_tensor3);

    tenGenTensor3.print_raw();
    tenGenTensor3.info();
}