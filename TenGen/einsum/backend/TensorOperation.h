#ifndef TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H
#define TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H

#include <cstdint>
#include <span>
#include <vector>

#include "TenGen.h"

using namespace TenGen::Types;
using namespace TenGen::Structs;
using namespace TenGen::MiniJit::Instructions::Encoding;
using Kernel = TenGen::MiniJit::Backend::Kernel;
using Util = TenGen::MiniJit::Generator::Util;

namespace TenGen::Einsum::Backend {

    class TensorOperation {
       public:
        // scalars
        dtype_t _dtype;
        prim_t _prim_first_touch;
        prim_t _prim_main;
        prim_t _prim_last_touch;

        // owned storage
        std::vector<dim_t> _dim_types_storage;
        std::vector<exec_t> _exec_types_storage;
        std::vector<int64_t> _dim_sizes_storage;
        std::vector<int64_t> _strides_in0_storage;
        std::vector<int64_t> _strides_in1_storage;
        std::vector<int64_t> _strides_out_storage;

        // views (spans)
        std::span<const dim_t> _dim_types;
        std::span<const exec_t> _exec_types;
        std::span<const int64_t> _dim_sizes;
        std::span<const int64_t> _strides_in0;
        std::span<const int64_t> _strides_in1;
        std::span<const int64_t> _strides_out;

        /**
         * Setup for a binary tensor contraction or a unary tensor operation.
         *
         * @param dtype             Datatype of all tensor elements.
         * @param prim_first_touch  Type of the first touch primitive.
         * @param prim_main         Type of the main primitive.
         * @param prim_last_touch   Type of the last touch primitive.
         * @param dim_types         Dimension type of the loops (c, m, n, or k).
         * @param exec_types        Execution type of the loops (seq, shared, or prim).
         * @param dim_sizes         Sizes of the dimensions.
         * @param strides_in0       Strides of the first input tensor.
         * @param strides_in1       Strides of the second input tensor (ignored if unary).
         * @param strides_out       Strides of the output tensor.
         * @return error_t::success on success, another error_t value otherwise.
         **/
        TenGen::Types::error_t setup(dtype_t dtype,
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

        /**
         * Execute the tensor operation.
         *
         * @param tensor_in0 First input tensor.
         * @param tensor_in1 Second input tensor (use nullptr if unary).
         * @param tensor_out Output tensor.
         **/
        void execute(void const* tensor_in0,
                     void const* tensor_in1,
                     void* tensor_out);

        /**
         * General-purpose loop implementation featuring first and last touch operations.
         * No threading is applied.
         *
         * @param id_loop      Dimension id of the loop which is executed.
         * @param ptr_in0      Pointer to the first input tensor's data.
         * @param ptr_in1      Pointer to the second input tensor's data (use nullptr if unary).
         * @param ptr_out      Pointer to the output tensor's data.
         * @param first_access True if first time accessing data of output tensor.
         * @param last_access  True if last time accessing data of output tensor.
         **/
        void execute_iter(int64_t id_loop,
                          char const* ptr_in0,
                          char const* ptr_in1,
                          char* ptr_out,
                          bool first_access,
                          bool last_access) {
            // int64_t l_size = m_loop_sizes[id_loop];

            // for (int64_t l_it = 0; l_it < l_size; l_it++) {
            //     // derive if this is first or last access to the output block

            //     // update pointer with strides

            //     if (id_loop + 1 < m_id_first_primitive_loop) {
            //         // recursive function call
            //     } else {
            //         // call first touch kernel if necessary

            //         // call main kernel

            //         // call last touch kernel if necessary
            //     }
            // }
        }
    };

}  // namespace TenGen::Einsum::Backend

#endif  // TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H