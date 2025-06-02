#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <iostream>
#include <span>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>

#include "TenGen.h"
#include "TenGen/einsum/backend/TensorOperation.h"
#include "TenGen/tensor/Tensor.h"
#include "TenGen/types/Structs.h"
#include "TenGen/types/Types.h"
#include "TenGenTests/TenGenTestsHelper.h"

using namespace TenGen::Types;
using namespace TenGen::Structs;
using TensorOperation = TenGen::Einsum::Backend::TensorOperation;

/*  Example 1 (einsum expression) abdc, ebfd -> aefc
    - dtype	FP32
    - prim_first_touch	None
    - prim_main	GEMM
    - prim_last_touch	None
    - dim_types  ( M, N, K, M, N, K )
    - exec_types ( Seq, Seq, Seq, Prim, Prim, Prim )
    - dim_sizes	 ( 32, 32, 8, 32, 32, 32 )
    - strides_in0	( 8192, 0, 1024, 1, 0, 32 )
    - strides_in1	( 0, 8192, 1024, 0, 32, 1 )
    - strides_out	( 32768, 1024, 0, 1, 32, 0 )

    aufbau input1: 32x8x32x32 -> abdc
    aufbau input2: 32x8x32x32 -> ebfd
    aufbau output: 32x32x32x32 -> aefc
*/

TEST_CASE("Example 1", "[Einsum][Backend][TensorOperation]") {
    // create a TenGen tensor and fill it with random values
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor1(true);
    TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor2(true);
    TenGen::Tensor<float, 32, 32, 32, 32> tenGenTensor3(true);
    TenGen::Tensor<float, 32, 32, 32, 32> tenGenTensor4;

    // copy data
    xt::xtensor<float, 4, xt::layout_type::column_major> xt_tensor1 = xt::xtensor<float, 4>::from_shape({32, 8, 32, 32});
    xt::xtensor<float, 4, xt::layout_type::column_major> xt_tensor2 = xt::xtensor<float, 4>::from_shape({32, 8, 32, 32});

    std::copy(tenGenTensor1.exportPointer(), tenGenTensor1.exportPointer() + tenGenTensor1.size, xt_tensor1.data());
    std::copy(tenGenTensor2.exportPointer(), tenGenTensor2.exportPointer() + tenGenTensor2.size, xt_tensor2.data());

    //
    //
    //

    TensorOperation to;

    // Setup the tensor operation
    auto l_error = to.setup(dtype_t::fp32,
                            prim_t::none,
                            prim_t::gemm,
                            prim_t::none,
                            std::vector<dim_t>{dim_t::m, dim_t::n, dim_t::k, dim_t::m, dim_t::n, dim_t::k},
                            std::vector<exec_t>{exec_t::seq, exec_t::seq, exec_t::seq, exec_t::prim, exec_t::prim, exec_t::prim},
                            std::vector<int64_t>{
                                tenGenTensor1.shape[0], tenGenTensor1.shape[1], tenGenTensor1.shape[2],
                                tenGenTensor2.shape[0], tenGenTensor2.shape[1], tenGenTensor2.shape[2]},
                            std::vector<int64_t>{tenGenTensor1.strides.begin(), tenGenTensor1.strides.end()}, std::vector<int64_t>{tenGenTensor2.strides.begin(), tenGenTensor2.strides.end()}, std::vector<int64_t>{tenGenTensor3.strides.begin(), tenGenTensor3.strides.end()});

    to.execute(tenGenTensor1.exportPointer(), tenGenTensor2.exportPointer(), tenGenTensor3.exportPointer());

    //
    //
    //

    // abdc, ebfd -> aefc
    auto result = xt::linalg::tensordot(xt_tensor1, xt_tensor2, std::vector<size_t>{1, 2}, std::vector<size_t>{1, 3});

    //
    //
    //

    std::copy(result.data(), result.data() + result.size(), tenGenTensor4.exportPointer());

    REQUIRE(tenGenTensor3 == tenGenTensor4);
}

/*  Example 2 (einsum expression) abdc, ebfd -> aefc
    - dtype	FP32
    - prim_first_touch	None
    - prim_main	GEMM
    - prim_last_touch	None
    - dim_types  ( M, N, K, M, N, K )
    - exec_types ( Seq, Seq, Seq, Prim, Prim, Prim )
    - dim_sizes	 ( 32, 32, 8, 32, 32, 32 )
    - strides_in0	( 8192, 0, 1024, 1, 0, 32 )
    - strides_in1	( 0, 8192, 1024, 0, 32, 1 )
    - strides_out	( 32768, 1024, 0, 1, 32, 0 )

    aufbau input1: 32x8x32x32 -> abdc
    aufbau input2: 32x8x32x32 -> ebfd
    aufbau output: 32x32x32x32 -> aefc
*/

// TEST_CASE("Example 2", "[Einsum][Backend][TensorOperation]") {
//     // create a TenGen tensor and fill it with random values
//     TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor1(fill_random = true);
//     TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor2(fill_random = true);
//     TenGen::Tensor<float, 32, 32, 32, 32> tenGenTensor3(fill_random = true);

//     std::span<const dim_t> l_dim_types = {dim_t::m, dim_t::n, dim_t::k, dim_t::m, dim_t::n, dim_t::k};
//     std::span<const exec_t> l_exec_types = {exec_t::seq, exec_t::seq, exec_t::seq, exec_t::prim, exec_t::prim, exec_t::prim};
//     std::span<const int64_t> l_dim_sizes = {tenGenTensor1.shape[0], tenGenTensor1.shape[1], tenGenTensor1.shape[2],
//                                             tenGenTensor2.shape[0], tenGenTensor2.shape[1], tenGenTensor2.shape[2]};

//     TensorConfig op{};

//     // Setup the tensor operation
//     auto l_error = setup(op,
//                          dtype_t::fp32,
//                          prim_t::none,
//                          prim_t::brgemm,
//                          prim_t::none,
//                          l_dim_types,
//                          l_exec_types,
//                          l_dim_sizes,
//                          tenGenTensor1.strides,
//                          tenGenTensor2.strides,
//                          tenGenTensor3.strides);
//     requires(l_error == error_t::success);

//     execute(op, tenGenTensor1, tenGenTensor2, tenGenTensor3);

//     // create a xtensor (real copy) from the TenGen tensor
//     auto xt_tensor1 = xt::adapt(tenGenTensor1.exportPointer(), tenGenTensor1.size(), xt::ownership(), {32, 8, 32, 32});
//     auto xt_tensor2 = xt::adapt(tenGenTensor2.exportPointer(), tenGenTensor2.size(), xt::ownership(), {32, 8, 32, 32});
//     // do einstein summation on both tensors with xtensor
//     auto xt_tensor3 = xt::linalg::einsum("mk,kn->mn", xt_tensor1, xt_tensor2);

//     TenGen::Tensor<float, 32, 32, 32, 32> tenGenTensor4;
//     tenGenTensor4.fromXTensor(xt_tensor3);
//     requires(tenGenTensor3 == tenGenTensor4);
// }

/*  Example 3 (einsum expression) abdc, ebfd -> aefc
    - dtype	FP32
    - prim_first_touch	None
    - prim_main	GEMM
    - prim_last_touch	None
    - dim_types  ( M, N, K, M, N, K )
    - exec_types ( Seq, Seq, Seq, Prim, Prim, Prim )
    - dim_sizes	 ( 32, 32, 8, 32, 32, 32 )
    - strides_in0	( 8192, 0, 1024, 1, 0, 32 )
    - strides_in1	( 0, 8192, 1024, 0, 32, 1 )
    - strides_out	( 32768, 1024, 0, 1, 32, 0 )

    aufbau input1: 32x8x32x32 -> abdc
    aufbau input2: 32x8x32x32 -> ebfd
    aufbau output: 32x32x32x32 -> aefc
*/

// TEST_CASE("Example 3", "[Einsum][Backend][TensorOperation]") {
//     // create a TenGen tensor and fill it with random values
//     TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor1(fill_random = true);
//     TenGen::Tensor<float, 32, 8, 32, 32> tenGenTensor2(fill_random = true);
//     TenGen::Tensor<float, 32, 32, 32, 32> tenGenTensor3(fill_random = true);

//     std::span<const dim_t> l_dim_types = {dim_t::m, dim_t::n, dim_t::k, dim_t::m, dim_t::n, dim_t::k};
//     std::span<const exec_t> l_exec_types = {exec_t::seq, exec_t::seq, exec_t::prim, exec_t::prim, exec_t::prim, exec_t::prim};
//     std::span<const int64_t> l_dim_sizes = {tenGenTensor1.shape[0], tenGenTensor1.shape[1], tenGenTensor1.shape[2],
//                                             tenGenTensor2.shape[0], tenGenTensor2.shape[1], tenGenTensor2.shape[2]};

//     TensorConfig op{};

//     // Setup the tensor operation
//     auto l_error = setup(op,
//                          dtype_t::fp32,
//                          prim_t::zero,
//                          prim_t::gemm,
//                          prim_t::relu,
//                          l_dim_types,
//                          l_exec_types,
//                          l_dim_sizes,
//                          tenGenTensor1.strides,
//                          tenGenTensor2.strides,
//                          tenGenTensor3.strides);
//     requires(l_error == error_t::success);

//     execute(op, tenGenTensor1, tenGenTensor2, tenGenTensor3);

//     // create a xtensor (real copy) from the TenGen tensor
//     auto xt_tensor1 = xt::adapt(tenGenTensor1.exportPointer(), tenGenTensor1.size(), xt::ownership(), {32, 8, 32, 32});
//     auto xt_tensor2 = xt::adapt(tenGenTensor2.exportPointer(), tenGenTensor2.size(), xt::ownership(), {32, 8, 32, 32});
//     // do einstein summation on both tensors with xtensor
//     auto xt_tensor3 = xt::linalg::einsum("mk,kn->mn", xt_tensor1, xt_tensor2);

//     TenGen::Tensor<float, 32, 32, 32, 32> tenGenTensor4;
//     tenGenTensor4.fromXTensor(xt_tensor3);
//     requires(tenGenTensor3 == tenGenTensor4);
// }