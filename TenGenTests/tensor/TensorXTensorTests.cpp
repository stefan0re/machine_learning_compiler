#include <catch2/catch_test_macros.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>

#include "../TenGenTestsHelper.h"
#include "TenGen.h"

TEST_CASE("Convert Tensor to xTensor", "[Tensor][Tensor][xTensor]") {
    // create a TenGen tensor and fill it with random values
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor1(true);

    xt::xtensor<float, 4> xt_tensor1 = xt::xtensor<float, 4>::from_shape({32, 8, 32, 32});

    // Copy the data from TenGen to xtensor
    std::copy(tenGenTensor1.exportPointer(), tenGenTensor1.exportPointer() + tenGenTensor1.size, xt_tensor1.data());
}

TEST_CASE("Calculations with xTensor", "[Tensor][Tensor][xTensor]") {
    // fill TenGen tensors
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor1(true);
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor2(true);

    // copy data
    xt::xtensor<float, 4> xt_tensor1 = xt::xtensor<float, 4>::from_shape({32, 8, 32, 32});
    xt::xtensor<float, 4> xt_tensor2 = xt::xtensor<float, 4>::from_shape({32, 8, 32, 32});

    std::copy(tenGenTensor1.exportPointer(), tenGenTensor1.exportPointer() + tenGenTensor1.size, xt_tensor1.data());
    std::copy(tenGenTensor2.exportPointer(), tenGenTensor2.exportPointer() + tenGenTensor2.size, xt_tensor2.data());

    // broadcasting logic
    auto input1_exp = xt::expand_dims(xt::expand_dims(xt_tensor1, 4), 5);   // shape: (32, 8, 32, 32, 1, 1)
    auto input2_perm = xt::transpose(xt_tensor2, {1, 3, 0, 2});             // shape: (8, 32, 32, 32)
    auto input2_exp = xt::expand_dims(xt::expand_dims(input2_perm, 3), 3);  // shape: (8, 32, 1, 1, 32, 32)

    // compute
    auto product = xt_tensor1 * input2_exp;
    auto result = xt::eval(xt::sum(product, {1, 2}));  // shape: (32, 32, 32, 32)
}

TEST_CASE("Calculations 2 with xTensor", "[Tensor][Tensor][xTensor]") {
    // fill TenGen tensors
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor1(true);
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor2(true);

    // copy data
    xt::xtensor<float, 4> xt_tensor1 = xt::xtensor<float, 4>::from_shape({32, 8, 32, 32});
    xt::xtensor<float, 4> xt_tensor2 = xt::xtensor<float, 4>::from_shape({32, 8, 32, 32});

    std::copy(tenGenTensor1.exportPointer(), tenGenTensor1.exportPointer() + tenGenTensor1.size, xt_tensor1.data());
    std::copy(tenGenTensor2.exportPointer(), tenGenTensor2.exportPointer() + tenGenTensor2.size, xt_tensor2.data());

    auto result = xt::linalg::tensordot(xt_tensor1, xt_tensor2, std::vector<size_t>{1, 2}, std::vector<size_t>{1, 3});
}
