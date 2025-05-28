#include "TensorOperation.h"

#include "../../mini_jit/generator/Unary.h"

using namespace mini_jit::generator;

namespace einsum::backend {

    TensorOperation::error_t TensorOperation::setup(dtype_t dtype,
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
        _dtype = dtype;
        _prim_first_touch = prim_first_touch;
        _prim_main = prim_main;
        _prim_last_touch = prim_last_touch;

        // safely copies all input arrays so they outlive setup()
        // 1. copy data into owned storage
        _dim_types_storage.assign(dim_types.begin(), dim_types.end());
        _exec_types_storage.assign(exec_types.begin(), exec_types.end());
        _dim_sizes_storage.assign(dim_sizes.begin(), dim_sizes.end());
        _strides_in0_storage.assign(strides_in0.begin(), strides_in0.end());
        _strides_in1_storage.assign(strides_in1.begin(), strides_in1.end());
        _strides_out_storage.assign(strides_out.begin(), strides_out.end());

        // 2. set spans to refer to owned data
        _dim_types = _dim_types_storage;
        _exec_types = _exec_types_storage;
        _dim_sizes = _dim_sizes_storage;
        _strides_in0 = _strides_in0_storage;
        _strides_in1 = _strides_in1_storage;
        _strides_out = _strides_out_storage;

        return error_t::success;
    }

}  // namespace einsum::backend