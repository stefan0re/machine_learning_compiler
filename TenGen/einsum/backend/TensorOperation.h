#ifndef TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H
#define TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H

#include <cstdint>
#include <span>
#include <vector>

#include "TenGen.h"

using namespace TenGen::Types;
using namespace TenGen::Structs;

namespace TenGen::Einsum::Backend::TensorOperation {

    // Function to initialize a TensorOperation
    TenGen::Types::error_t setup(TensorConfig& op,
                                 dtype_t dtype,
                                 prim_t prim_first_touch,
                                 prim_t prim_main,
                                 prim_t prim_last_touch,
                                 std::span<const dim_t> dim_types,
                                 std::span<const exec_t> exec_types,
                                 std::span<const int64_t> dim_sizes,
                                 std::span<const int64_t> strides_in0,
                                 std::span<const int64_t> strides_in1,
                                 std::span<const int64_t> strides_out) {
        // Store scalars
        op.dtype = dtype;
        op.prim_first_touch = prim_first_touch;
        op.prim_main = prim_main;
        op.prim_last_touch = prim_last_touch;

        // Copy data into owned storage
        op.dim_types_storage.assign(dim_types.begin(), dim_types.end());
        op.exec_types_storage.assign(exec_types.begin(), exec_types.end());
        op.dim_sizes_storage.assign(dim_sizes.begin(), dim_sizes.end());
        op.strides_in0_storage.assign(strides_in0.begin(), strides_in0.end());
        op.strides_in1_storage.assign(strides_in1.begin(), strides_in1.end());
        op.strides_out_storage.assign(strides_out.begin(), strides_out.end());

        // Set spans to refer to owned data
        op.dim_types = op.dim_types_storage;
        op.exec_types = op.exec_types_storage;
        op.dim_sizes = op.dim_sizes_storage;
        op.strides_in0 = op.strides_in0_storage;
        op.strides_in1 = op.strides_in1_storage;
        op.strides_out = op.strides_out_storage;

        return error_t::success;
    }

    // Function to execute a tensor operation
    void execute(const TensorConfig& op,
                 const void* tensor_in0,
                 const void* tensor_in1,
                 void* tensor_out) {
        // Implementation to be filled in
    }

    // Recursive loop execution function
    void execute_iter(const TensorConfig& op,
                      int64_t id_loop,
                      const char* ptr_in0,
                      const char* ptr_in1,
                      char* ptr_out,
                      bool first_access,
                      bool last_access) {
        // Placeholder for future implementation
    }

}  // namespace TenGen::Einsum::Backend::TensorOperation

#endif  // TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H
