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

        if (_prim_last_touch == prim_t::relu) {
            _is_last_touch_relu = true;
        } else {
            _is_last_touch_relu = false;
        }

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

        // Check if the first loop should be executed in parallel
        bool use_parallel = false;
        if (_loop_ids.size() > 0 && _exec_types[_loop_ids[0]] == exec_t::shared) {
            use_parallel = true;
        }

        if (use_parallel) {
            execute_iter_parallel(0, l_ptr_in0, l_ptr_in1, l_ptr_out, false, false);
        } else {
            execute_iter(0, l_ptr_in0, l_ptr_in1, l_ptr_out, false, false);
        }
    }
    void TensorOperation::execute_iter(int64_t id_loop,
                                       char const* ptr_in0,
                                       char const* ptr_in1,
                                       char* ptr_out,
                                       bool first_access,
                                       bool last_access) {
        int64_t l_size = 1;
        if (_loop_ids.size() > 0) {
            l_size = _dim_sizes[_loop_ids[id_loop]];
        }

        for (int64_t l_it = 0; l_it < l_size; l_it++) {
            // derive if this is first or last access to the output block
            if (id_loop == 0 && _dim_types[_loop_ids[id_loop]] != dim_t::k && _loop_ids.size() > 0) {
                first_access = true;
            } else if (id_loop == 0 && _dim_types[_loop_ids[id_loop]] == dim_t::k && l_it == 0 && _loop_ids.size() > 0) {
                first_access = true;
            } else if (_loop_ids.size() == 0) {
                first_access = true;
            } else {
                first_access = false;
            }
            if ((id_loop == _loop_ids.size() - 1) && (_dim_types[_loop_ids[id_loop]] != dim_t::k)) {
                last_access = true;
            } else if ((id_loop == _loop_ids.size() - 1) && (_dim_types[_loop_ids[id_loop]] == dim_t::k) && (l_it == l_size - 1)) {
                last_access = true;
            } else if (_loop_ids.size() == 0) {
                last_access = true;
            } else {
                last_access = false;
            }

            // update pointer with strides
            char* l_ptr_in0 = const_cast<char*>(ptr_in0);
            char* l_ptr_in1 = const_cast<char*>(ptr_in1);
            char* l_ptr_out = ptr_out;

            if (_loop_ids.size() > 0) {
                l_ptr_in0 += l_it * _strides_in0[_loop_ids[id_loop]] * 4;
                l_ptr_in1 += l_it * _strides_in1[_loop_ids[id_loop]] * 4;
                l_ptr_out += l_it * _strides_out[_loop_ids[id_loop]] * 4;
            }
            if ((_loop_ids.size() > 0) && (id_loop < _loop_ids.size() - 1)) {
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
                if (_is_last_touch_relu && last_access) {
                    _brgemm_last_touch_kernel(l_ptr_in0,
                                              l_ptr_in1,
                                              l_ptr_out,
                                              _lda,
                                              _ldb,
                                              _ldc,
                                              _br_stride_a,
                                              _br_stride_b);
                } else {
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
    }

    void TensorOperation::execute_iter_parallel(int64_t id_loop,
                                                const char* ptr_in0,
                                                const char* ptr_in1,
                                                char* ptr_out,
                                                bool first_access,
                                                bool last_access) {
        int64_t l_size = 1;
        if (_loop_ids.size() > 0) {
            l_size = _dim_sizes[_loop_ids[id_loop]];
        }

#pragma omp parallel for
        for (int64_t l_it = 0; l_it < l_size; l_it++) {
            // derive if this is first or last access to the output block
            bool local_first_access = first_access;
            bool local_last_access = false;

            if (id_loop == 0) {
                local_first_access = true;
            }
            if ((id_loop == _loop_ids.size() - 1) && (_dim_types[_loop_ids[id_loop]] != dim_t::k)) {
                local_last_access = true;
            } else if ((id_loop == _loop_ids.size() - 1) && (_dim_types[_loop_ids[id_loop]] == dim_t::k) && (l_it == l_size - 1)) {
                local_last_access = true;
            }

            // update pointer with strides
            char* l_ptr_in0 = const_cast<char*>(ptr_in0);
            char* l_ptr_in1 = const_cast<char*>(ptr_in1);
            char* l_ptr_out = ptr_out;

            if (_loop_ids.size() > 0) {
                l_ptr_in0 += l_it * _strides_in0[_loop_ids[id_loop]] * 4;
                l_ptr_in1 += l_it * _strides_in1[_loop_ids[id_loop]] * 4;
                l_ptr_out += l_it * _strides_out[_loop_ids[id_loop]] * 4;
            }

            if ((_loop_ids.size() > 0) && (id_loop < _loop_ids.size() - 1)) {
                // recursive function call (sequential from here)
                execute_iter(id_loop + 1,
                             l_ptr_in0,
                             l_ptr_in1,
                             l_ptr_out,
                             false,
                             false);
            } else {
                // call first touch kernel if necessary
                if (local_first_access && _prim_first_touch != prim_t::none) {
                    _unary_first_touch_kernel(l_ptr_out, l_ptr_out, _ldc, _ldc);
                }
                if (_is_last_touch_relu && local_last_access) {
                    _brgemm_last_touch_kernel(l_ptr_in0,
                                              l_ptr_in1,
                                              l_ptr_out,
                                              _lda,
                                              _ldb,
                                              _ldc,
                                              _br_stride_a,
                                              _br_stride_b);
                } else {
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
                    if (local_last_access && _prim_last_touch != prim_t::none) {
                        _unary_last_touch_kernel(l_ptr_out, l_ptr_out, _ldc, _ldc);
                    }
                }
            }
        }
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
        if (_loop_ids.size() == 0) {
            for (size_t i = 0; i < _exec_types.size(); i++) {
                if (_exec_types[i] == exec_t::seq || _exec_types[i] == exec_t::shared) {
                    _loop_ids.push_back(i);
                }
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
                         static_cast<mini_jit::generator::Brgemm::dtype_t>(_dtype),
                         false);
        _brgemm_kernel = _brgemm.get_kernel();

        if (_is_last_touch_relu) {
            _brgemm_last_touch.generate(_dim_sizes[_id_prim_m],
                                        _dim_sizes[_id_prim_n],
                                        _dim_sizes[_id_prim_k],
                                        (_id_prim_br != -1) ? _dim_sizes[_id_prim_br] : 1,
                                        0,
                                        0,
                                        0,
                                        static_cast<mini_jit::generator::Brgemm::dtype_t>(_dtype),
                                        true);
        }
        _brgemm_last_touch_kernel = _brgemm_last_touch.get_kernel();

        // generate first/last touch primitive
        if (!(_prim_first_touch == prim_t::none)) {
            _unary_first_touch.generate(_dim_sizes[_id_prim_m],
                                        _dim_sizes[_id_prim_n],
                                        static_cast<mini_jit::generator::Unary::dtype_t>(_dtype),
                                        static_cast<mini_jit::generator::Unary::ptype_t>(_prim_first_touch));
            _unary_first_touch_kernel = _unary_first_touch.get_kernel();
        }
        if (!(_prim_last_touch == prim_t::none) && !_is_last_touch_relu) {
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

    /********************************************/
    /** IR for optimizations on TensorOperation */
    /********************************************/

    TensorOperation::error_t TensorOperation::optimize() {
        fuse_dimensions();
        split_dimensions();
        identify_primitives();
        reorder_dimensions();
        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::split_dimensions() {
        // split M dimension if larger than 128
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::m && _dim_sizes[i] > 128) {
                std::vector<int64_t> pf = prime_factors(_dim_sizes[i]);
                int64_t split_size_0 = find_new_size(pf);
                int64_t split_size_1 = _dim_sizes[i] / split_size_0;
                if (split_size_0 == 1 || split_size_1 == 1) {
                    continue;  // no split possible
                }
                // refactor dimension i
                _dim_sizes[i] = split_size_0;

                // add new dimension
                _dim_types.insert(_dim_types.begin() + i + 1, dim_t::m);
                _exec_types.insert(_exec_types.begin() + i + 1, exec_t::seq);
                _dim_sizes.insert(_dim_sizes.begin() + i + 1, split_size_1);
                _strides_in0.insert(_strides_in0.begin() + i + 1, _strides_in0[i] * split_size_0);
                _strides_in1.insert(_strides_in1.begin() + i + 1, _strides_in1[i] * split_size_0);
                _strides_out.insert(_strides_out.begin() + i + 1, _strides_out[i] * split_size_0);
            }
        }

        // split N dimension if larger than 128
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::n && _dim_sizes[i] > 128) {
                std::vector<int64_t> pf = prime_factors(_dim_sizes[i]);
                int64_t split_size_0 = find_new_size(pf);
                int64_t split_size_1 = _dim_sizes[i] / split_size_0;
                if (split_size_0 == 1 || split_size_1 == 1) {
                    continue;  // no split possible
                }
                // refactor dimension i
                _dim_sizes[i] = split_size_0;

                // add new dimension
                _dim_types.insert(_dim_types.begin() + i + 1, dim_t::n);
                _exec_types.insert(_exec_types.begin() + i + 1, exec_t::seq);
                _dim_sizes.insert(_dim_sizes.begin() + i + 1, split_size_1);
                _strides_in0.insert(_strides_in0.begin() + i + 1, _strides_in0[i] * split_size_0);
                _strides_in1.insert(_strides_in1.begin() + i + 1, _strides_in1[i] * split_size_0);
                _strides_out.insert(_strides_out.begin() + i + 1, _strides_out[i] * split_size_0);
            }
        }

        // split K dimension if larger than 128
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::k && _dim_sizes[i] > 128) {
                std::vector<int64_t> pf = prime_factors(_dim_sizes[i]);
                int64_t split_size_0 = find_new_size(pf);
                int64_t split_size_1 = _dim_sizes[i] / split_size_0;
                if (split_size_0 == 1 || split_size_1 == 1) {
                    continue;  // no split possible
                }
                // refactor dimension i
                _dim_sizes[i] = split_size_0;

                // add new dimension
                _dim_types.insert(_dim_types.begin() + i + 1, dim_t::k);
                _exec_types.insert(_exec_types.begin() + i + 1, exec_t::seq);
                _dim_sizes.insert(_dim_sizes.begin() + i + 1, split_size_1);
                _strides_in0.insert(_strides_in0.begin() + i + 1, _strides_in0[i] * split_size_0);
                _strides_in1.insert(_strides_in1.begin() + i + 1, _strides_in1[i] * split_size_0);
                _strides_out.insert(_strides_out.begin() + i + 1, _strides_out[i] * split_size_0);
            }
        }

        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::fuse_dimensions() {
        // fuse M dimension if smaller than 64
        for (size_t i = 1; i < _dim_types.size() - 1; i++) {
            if (_dim_types[i] == dim_t::m && _dim_sizes[i] < 64) {
                int64_t tmp_stride_in0 = _strides_in0[i];
                int64_t tmp_stride_out = _strides_out[i];
                int64_t tmp_dim_size = _dim_sizes[i];
                for (size_t j = 0; j < _dim_sizes.size(); j++) {
                    // fuse with smaller stride
                    if ((tmp_stride_in0 == _strides_in0[j] * tmp_dim_size) && (tmp_stride_out == _strides_out[j] * tmp_dim_size)) {
                        // fuse dimensions
                        _dim_sizes[j] *= tmp_dim_size;
                        // remove dimension i
                        _dim_types.erase(_dim_types.begin() + i);
                        _exec_types.erase(_exec_types.begin() + i);
                        _dim_sizes.erase(_dim_sizes.begin() + i);
                        _strides_in0.erase(_strides_in0.begin() + i);
                        _strides_in1.erase(_strides_in1.begin() + i);
                        _strides_out.erase(_strides_out.begin() + i);
                    }  // fuse with bigger stride
                    else if (tmp_stride_in0 * _dim_sizes[i] == _strides_in0[j] && tmp_stride_out * _dim_sizes[i] == _strides_out[j]) {
                        // fuse dimensions
                        _dim_sizes[j] *= tmp_dim_size;
                        _strides_in0[j] = _strides_in0[i];
                        _strides_out[j] = _strides_out[i];
                        // remove dimension i
                        _dim_types.erase(_dim_types.begin() + i);
                        _exec_types.erase(_exec_types.begin() + i);
                        _dim_sizes.erase(_dim_sizes.begin() + i);
                        _strides_in0.erase(_strides_in0.begin() + i);
                        _strides_in1.erase(_strides_in1.begin() + i);
                        _strides_out.erase(_strides_out.begin() + i);
                    }
                }
            }
        }

        // fuse N dimension if smaller than 64
        for (size_t i = 1; i < _dim_types.size() - 1; i++) {
            if (_dim_types[i] == dim_t::n && _dim_sizes[i] < 64) {
                int64_t tmp_stride_in1 = _strides_in1[i];
                int64_t tmp_stride_out = _strides_out[i];
                int64_t tmp_dim_size = _dim_sizes[i];
                for (size_t j = 0; j < _dim_sizes.size(); j++) {
                    // fuse with smaller stride
                    if ((tmp_stride_in1 == _strides_in1[j] * tmp_dim_size) && (tmp_stride_out == _strides_out[j] * tmp_dim_size)) {
                        // fuse dimensions
                        _dim_sizes[j] *= tmp_dim_size;
                        // remove dimension i
                        _dim_types.erase(_dim_types.begin() + i);
                        _exec_types.erase(_exec_types.begin() + i);
                        _dim_sizes.erase(_dim_sizes.begin() + i);
                        _strides_in0.erase(_strides_in0.begin() + i);
                        _strides_in1.erase(_strides_in1.begin() + i);
                        _strides_out.erase(_strides_out.begin() + i);
                    }  // fuse with bigger stride
                    else if (tmp_stride_in1 * _dim_sizes[i] == _strides_in1[j] && tmp_stride_out * _dim_sizes[i] == _strides_out[j]) {
                        // fuse dimensions
                        _dim_sizes[j] *= tmp_dim_size;
                        _strides_in1[j] = _strides_in1[i];
                        _strides_out[j] = _strides_out[i];
                        // remove dimension i
                        _dim_types.erase(_dim_types.begin() + i);
                        _exec_types.erase(_exec_types.begin() + i);
                        _dim_sizes.erase(_dim_sizes.begin() + i);
                        _strides_in0.erase(_strides_in0.begin() + i);
                        _strides_in1.erase(_strides_in1.begin() + i);
                        _strides_out.erase(_strides_out.begin() + i);
                    }
                }
            }
        }

        // fuse K dimension if smaller than 64
        for (size_t i = 1; i < _dim_types.size() - 1; i++) {
            if (_dim_types[i] == dim_t::k && _dim_sizes[i] < 64) {
                int64_t tmp_stride_in0 = _strides_in0[i];
                int64_t tmp_stride_in1 = _strides_in1[i];
                int64_t tmp_dim_size = _dim_sizes[i];
                for (size_t j = 0; j < _dim_sizes.size(); j++) {
                    // fuse with smaller stride
                    if ((tmp_stride_in0 == _strides_in0[j] * tmp_dim_size) && (tmp_stride_in1 == _strides_in1[j] * tmp_dim_size)) {
                        // fuse dimensions
                        _dim_sizes[j] *= tmp_dim_size;
                        // remove dimension i
                        _dim_types.erase(_dim_types.begin() + i);
                        _exec_types.erase(_exec_types.begin() + i);
                        _dim_sizes.erase(_dim_sizes.begin() + i);
                        _strides_in0.erase(_strides_in0.begin() + i);
                        _strides_in1.erase(_strides_in1.begin() + i);
                        _strides_out.erase(_strides_out.begin() + i);
                    }  // fuse with bigger stride
                    else if (tmp_stride_in0 * tmp_dim_size == _strides_in0[j] && tmp_stride_in1 * tmp_dim_size == _strides_in1[j]) {
                        // fuse dimensions
                        _dim_sizes[j] *= tmp_dim_size;
                        _strides_in0[j] = _strides_in0[i];
                        _strides_in1[j] = _strides_in1[i];
                        // remove dimension i
                        _dim_types.erase(_dim_types.begin() + i);
                        _exec_types.erase(_exec_types.begin() + i);
                        _dim_sizes.erase(_dim_sizes.begin() + i);
                        _strides_in0.erase(_strides_in0.begin() + i);
                        _strides_in1.erase(_strides_in1.begin() + i);
                        _strides_out.erase(_strides_out.begin() + i);
                    }
                }
            }
        }
        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::reorder_dimensions() {
        // seperate Dimensions
        std::vector<int64_t> m_loops;
        std::vector<int64_t> n_loops;
        std::vector<int64_t> k_loops;

        for (size_t i = 0; i < _exec_types.size(); i++) {
            if (_exec_types[i] == exec_t::seq || _exec_types[i] == exec_t::shared) {
                if (_dim_types[i] == dim_t::m) {
                    m_loops.push_back(i);
                } else if (_dim_types[i] == dim_t::n) {
                    n_loops.push_back(i);
                } else if (_dim_types[i] == dim_t::k) {
                    k_loops.push_back(i);
                }
            }
        }

        // assign loop dimensions to _loop_ids vector
        size_t max_loops_per_dim = std::max({m_loops.size(), n_loops.size(), k_loops.size()});
        for (size_t i = 0; i < max_loops_per_dim; i++) {
            if (i < m_loops.size()) {
                _loop_ids.push_back(m_loops[i]);
            }
            if (i < n_loops.size()) {
                _loop_ids.push_back(n_loops[i]);
            }
            if (i < k_loops.size()) {
                _loop_ids.push_back(k_loops[i]);
            }
        }
        if (_loop_ids.size() > 0 && _dim_types[_loop_ids[0]] != dim_t::k) {
            _id_parallel_loop = _loop_ids[0];
            _exec_types[_id_parallel_loop] = exec_t::shared;
        }

        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::identify_primitives() {
        // identify prim M
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::m && _strides_in0[i] == 1 && _strides_out[i] == 1) {
                _id_prim_m = i;
                _exec_types[i] = exec_t::prim;
                break;
            }
        }

        // identify prim N
        int64_t smallest_stride = 1e18;
        for (size_t i = 0; i < _dim_types.size(); i++) {
            // find smallest stride in N dimension
            if (_dim_types[i] == dim_t::n && _strides_in1[i] > 0) {
                smallest_stride = std::min(smallest_stride, _strides_in1[i]);
            }
        }
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (smallest_stride == _strides_in1[i] && _dim_types[i] == dim_t::n) {
                _id_prim_n = i;
                _exec_types[i] = exec_t::prim;
                break;
            }
        }

        // identify prim K
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::k && _strides_in1[i] == 1) {
                _id_prim_k = i;
                _exec_types[i] = exec_t::prim;
                break;
            }
        }

        // identify prim BR
        smallest_stride = 1e18;
        for (size_t i = 0; i < _dim_types.size(); i++) {
            // find smalles stride in BR dimension
            if (_dim_types[i] == dim_t::k && _strides_in1[i] > 1) {
                smallest_stride = std::min(smallest_stride, _strides_in1[i]);
            }
        }
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (smallest_stride == _strides_in1[i] && _dim_types[i] == dim_t::k) {
                if (_dim_sizes[i] > 16) {
                    continue;  // skip if size is 1
                }
                _id_prim_br = i;
                _exec_types[i] = exec_t::prim;
                break;
            }
        }

        // check if all ids are set
        if (_id_prim_m == -1 || _id_prim_n == -1 || _id_prim_k == -1) {
            std::cerr << "Error: Not all primitive ids are set correctly." << std::endl;
            return TensorOperation::error_t::compile_failed;
        }

        return TensorOperation::error_t::success;
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

}  // namespace einsum::backend
