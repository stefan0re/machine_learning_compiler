#include <catch2/catch_test_macros.hpp>

#include "Fastor/Fastor.h"
#include "TenGen.h"
#include "TenGenTests/TenGenTestsHelper.h"

using namespace Fastor;

enum { M,
       N,
       K };

TEST_CASE("Fastor Tensor == TenGen Tensor", "[Tensor][Tensor]") {
    // create a TenGen tensor and fill it with random values
    TenGen::Tensor<float, 2, 2> tenGenTensor1;
    tenGenTensor1.fillRandom();
    TenGen::Tensor<float, 2, 2> tenGenTensor2;
    tenGenTensor2.fillRandom();

    // create a Fastor tensor (real copy) from the TenGen tensor
    Fastor::Tensor<float, 2, 2> fastorTensor1(tenGenTensor1.exportPointer());
    Fastor::Tensor<float, 2, 2> fastorTensor2(tenGenTensor2.exportPointer());

    // do einstein summation on both tensors with fastor
    auto C = einsum<Index<M, K>, Index<K, N>>(fastorTensor1, fastorTensor2);

    TenGen::Tensor<float, 2, 2> tenGenTensor3;
    tenGenTensor3.fromFastor(C);

    tenGenTensor3.print_raw();
    tenGenTensor3.info();
}