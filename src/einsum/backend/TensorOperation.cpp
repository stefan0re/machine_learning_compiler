#include "TensorOperation.h"

#include <omp.h>

#include <algorithm>
#include <iostream>

#include "../../tensor/tensor.h"
#include "../include/einsum_ref.h"

// #define DEBUG

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
        // set primitive types and dtype
        _prim_first_touch = prim_first_touch;
        _prim_main = prim_main;
        _prim_last_touch = prim_last_touch;
        _dtype = dtype;

        // set vectors
        _dim_types.assign(dim_types.begin(), dim_types.end());
        _exec_types.assign(exec_types.begin(), exec_types.end());
        _dim_sizes.assign(dim_sizes.begin(), dim_sizes.end());
        _strides_in0.assign(strides_in0.begin(), strides_in0.end());
        _strides_in1.assign(strides_in1.begin(), strides_in1.end());
        _strides_out.assign(strides_out.begin(), strides_out.end());

        return TensorOperation::error_t::success;
    }
    void TensorOperation::execute(void const* tensor_in0,
                                  void const* tensor_in1,
                                  void* tensor_out) {
        // get pointers to input and output data
        char const* l_ptr_in0 = static_cast<char const*>(tensor_in0);
        char const* l_ptr_in1 = static_cast<char const*>(tensor_in1);
        char* l_ptr_out = static_cast<char*>(tensor_out);

        execute_iter(0, l_ptr_in0, l_ptr_in1, l_ptr_out, false, false);
    }
    void TensorOperation::execute_iter(int64_t id_loop,
                                       char const* ptr_in0,
                                       char const* ptr_in1,
                                       char* ptr_out,
                                       bool first_access,
                                       bool last_access) {
        int64_t l_size = _dim_sizes[_loop_ids[id_loop]];

        for (int64_t l_it = 0; l_it < l_size; l_it++) {
            // derive if this is first or last access to the output block
            if (l_it == 0 && id_loop == 0) {
                first_access = true;
            }
            if ((id_loop == _loop_ids.size() - 1) && (_dim_types[_loop_ids[id_loop]] != dim_t::k)) {
                last_access = true;
            } else if ((id_loop == _loop_ids.size() - 1) && (_dim_types[_loop_ids[id_loop]] == dim_t::k) && (l_it == l_size - 1)) {
                last_access = true;
            } else {
                last_access = false;
            }

            // update pointer with strides
            char* l_ptr_in0 = const_cast<char*>(ptr_in0);
            char* l_ptr_in1 = const_cast<char*>(ptr_in1);
            char* l_ptr_out = ptr_out;

            l_ptr_in0 += l_it * _strides_in0[_loop_ids[id_loop]] * 4;
            l_ptr_in1 += l_it * _strides_in1[_loop_ids[id_loop]] * 4;
            l_ptr_out += l_it * _strides_out[_loop_ids[id_loop]] * 4;

            if (id_loop < _loop_ids.size() - 1) {
                // recursive function call
                execute_iter(id_loop + 1,
                             l_ptr_in0,
                             l_ptr_in1,
                             l_ptr_out,
                             false,
                             false);
            } else {
                // call first touch kernel if necessary
                if (first_access && _prim_first_touch != prim_t::none) {
                    _unary_first_touch_kernel(l_ptr_out, l_ptr_out, _ldc, _ldc);
                }
                // call main kernel
                _brgemm_kernel(l_ptr_in0,
                               l_ptr_in1,
                               l_ptr_out,
                               _lda,
                               _ldb,
                               _ldc,
                               _br_stride_a,
                               _br_stride_b);

                // call last touch kernel if necessary
                if (last_access && _prim_last_touch != prim_t::none) {
                    _unary_last_touch_kernel(l_ptr_out, l_ptr_out, _ldc, _ldc);
                }
            }
        }
    }

    TensorOperation::error_t TensorOperation::optimize() {
        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::split_dimensions() {
        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::fuse_dimensions() {
        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::reorder_dimensions() {
        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::identify_primitives() {
        return TensorOperation::error_t::success;
    }

    /** The folowing is set here:
     * - First touch primitive
     * - Main primitive
     * - Last touch primitive
     * - Loop vector
     * - Primitive ids
     * - Runtime parameters
     */
    TensorOperation::error_t TensorOperation::compile() {
        // Initialize loop_ids
        for (size_t i = 0; i < _exec_types.size(); i++) {
            if (_exec_types[i] == exec_t::seq || _exec_types[i] == exec_t::shared) {
                _loop_ids.push_back(i);
            }
        }

        // initialize id_prims
        _id_prim_m = -1;
        _id_prim_n = -1;
        _id_prim_k = -1;
        _id_prim_br = -1;

        for (size_t i = 0; i < _exec_types.size(); i++) {
            if (_exec_types[i] == exec_t::prim) {
                if (_dim_types[i] == dim_t::m) {
                    _id_prim_m = i;
                } else if (_dim_types[i] == dim_t::n) {
                    _id_prim_n = i;
                }
            }
        }

        // if k is stride 1 in in1 then K, else BR
        for (size_t i = 0; i < _exec_types.size(); i++) {
            if (_exec_types[i] == exec_t::prim && _dim_types[i] == dim_t::k) {
                if (_strides_in1[i] == 1) {
                    _id_prim_k = i;
                } else {
                    _id_prim_br = i;
                }
            }
        }
        // check if all ids are set
        if (_id_prim_m == -1 || _id_prim_n == -1 || _id_prim_k == -1) {
            std::cerr << "Error: Not all primitive ids are set correctly." << std::endl;
            return TensorOperation::error_t::compile_failed;
        }

        // generate main primitive
        _brgemm.generate(_dim_sizes[_id_prim_m],
                         _dim_sizes[_id_prim_n],
                         _dim_sizes[_id_prim_k],
                         (_id_prim_br != -1) ? _dim_sizes[_id_prim_br] : 1,
                         0,
                         0,
                         0,
                         static_cast<mini_jit::generator::Brgemm::dtype_t>(_dtype));
        _brgemm_kernel = _brgemm.get_kernel();

        // generate first/last touch primitive
        if (!(_prim_first_touch == prim_t::none)) {
            _unary_first_touch.generate(_dim_sizes[_id_prim_m],
                                        _dim_sizes[_id_prim_n],
                                        static_cast<mini_jit::generator::Unary::dtype_t>(_dtype),
                                        static_cast<mini_jit::generator::Unary::ptype_t>(_prim_first_touch));
            _unary_first_touch_kernel = _unary_first_touch.get_kernel();
        }
        if (!(_prim_last_touch == prim_t::none)) {
            _unary_last_touch.generate(_dim_sizes[_id_prim_m],
                                       _dim_sizes[_id_prim_n],
                                       static_cast<mini_jit::generator::Unary::dtype_t>(_dtype),
                                       static_cast<mini_jit::generator::Unary::ptype_t>(_prim_last_touch));
            _unary_last_touch_kernel = _unary_last_touch.get_kernel();
        }

        // set runtime parameter
        _lda = _strides_in0[_id_prim_k];
        _ldb = _strides_in1[_id_prim_n];
        _ldc = _strides_out[_id_prim_n];
        if (_id_prim_br != -1) {
            _br_stride_a = _strides_in0[_id_prim_br];
            _br_stride_b = _strides_in1[_id_prim_br];
        }

        // check if relevant runtime parameter are set
        if (_lda == 0 || _ldb == 0 || _ldc == 0) {
            std::cerr << "Error: Leading dimensions cannot be zero." << std::endl;
            return TensorOperation::error_t::compile_failed;
        }
        if (_br_stride_a < 0 || _br_stride_b < 0) {
            std::cerr << "Error: Batch-reduce strides cannot be negative." << std::endl;
            return TensorOperation::error_t::compile_failed;
        }

        return TensorOperation::error_t::success;
    }

    void TensorOperation::execute_iter_parallel(const char* ptr_in0,
                                                const char* ptr_in1,
                                                const char* ptr_bias,
                                                char* ptr_out,
                                                bool first_access,
                                                bool last_access) {
    }

    int64_t TensorOperation::get_flops_count() {
        int64_t flops = 2;
        for (size_t i = 0; i < _dim_sizes.size(); i++) {
            flops *= _dim_sizes[i];
        }
        int64_t minus = 1;
        for (size_t i = 0; i < _dim_sizes.size(); i++) {
            if (_dim_types[i] == dim_t::m || _dim_types[i] == dim_t::n) {
                minus *= _dim_sizes[i];
            }
        }
        return flops - minus;
    }

    void TensorOperation::print() {
    }
}  // namespace einsum::backend
