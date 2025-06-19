#include "TensorOperation.h"

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
                                                    Tensor& in0,
                                                    Tensor& in1,
                                                    Tensor& out) {
        _dtype = dtype;
        _prim_first_touch = prim_first_touch;
        _prim_main = prim_main;
        _prim_last_touch = prim_last_touch;

        _tensor_in0 = &in0;
        _tensor_in1 = &in1;
        _tensor_out = &out;

        return TensorOperation::error_t::success;
    }
    void TensorOperation::execute(void const* tensor_in0,
                                  void const* tensor_in1,
                                  void* tensor_out) {
        // get pointers to input and output data
        char const* l_ptr_in0 = static_cast<char const*>(tensor_in0);
        char const* l_ptr_in1 = static_cast<char const*>(tensor_in1);
        char* l_ptr_out = static_cast<char*>(tensor_out);

        // execute the operation
        execute_iter_parallel(l_ptr_in0, l_ptr_in1, l_ptr_out, false, false);
    }
    void TensorOperation::execute_iter(int64_t id_loop,
                                       char const* ptr_in0,
                                       char const* ptr_in1,
                                       char* ptr_out,
                                       bool first_access,
                                       bool last_access) {
        int64_t l_size = _tensor_in0->id[_loop_order[id_loop]].dim_sizes;

        for (int64_t l_it = 0; l_it < l_size; l_it++) {
            // derive if this is first or last access to the output block

            // update pointer with strides
            char* l_ptr_in0 = const_cast<char*>(ptr_in0) + l_it * _tensor_in0->id[_loop_order[id_loop]].stride * 4;
            char* l_ptr_in1 = const_cast<char*>(ptr_in1) + l_it * _tensor_in1->id[_loop_order[id_loop]].stride * 4;
            char* l_ptr_out = ptr_out + l_it * _tensor_out->id[_loop_order[id_loop]].stride * 4;

            if (id_loop + 1 < _id_first_primitive_loop) {
                execute_iter(id_loop + 1,
                             ptr_in0,
                             ptr_in1,
                             ptr_out,
                             first_access,
                             last_access);
            } else {
                // call first touch kernel if necessary

                // call main kernel
                _brgemm_kernel(l_ptr_in0,
                               l_ptr_in1,
                               l_ptr_out,
                               _lda,
                               _ldb,
                               _ldc,
                               0,
                               0);

                // call last touch kernel if necessary
            }
        }
    }

    TensorOperation::error_t TensorOperation::optimize() {
        std::cout << "dimension sizes: "
                  << _tensor_in0->id.size() << ", "
                  << _tensor_in1->id.size() << ", "
                  << _tensor_out->id.size() << std::endl;

        // check if all three tensor dimension vectors (id) have the same size
        if (!(_tensor_in0->id.size() == _tensor_in1->id.size() && _tensor_out->id.size() == _tensor_in0->id.size())) {
            std::cerr << "Dimension sizes must be equal" << std::endl;
            return TensorOperation::error_t::optimize_failed;
        }

        // fuse_dimensions();
        // split_dimensions();
        identify_primitives();
        reorder_dimensions();

        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::split_dimensions() {
        // fuse M dimension in input tensor 0 and output tensor

        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::fuse_dimensions() {
        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::reorder_dimensions() {
        // reoder loop dimension to a M N M N pattern if it is possible

        // check if last loop is M or N than make it shared
        if (_loop_order.size() > 0) {
            if (_tensor_in0->id[_loop_order.back()].dim_t == 1 || _tensor_in1->id[_loop_order.back()].dim_t == 2) {
                _tensor_in0->id[_loop_order.back()].exec_t = 2;
                _tensor_in1->id[_loop_order.back()].exec_t = 2;
                _tensor_out->id[_loop_order.back()].exec_t = 2;
            }
        }

        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::identify_primitives() {
        bool found_m, found_n, found_k = false;

        // set M and K Primitve
        for (size_t i = 0; i < _tensor_in0->id.size(); i++) {
            if (_tensor_in0->id[i].dim_t == 1 && _tensor_out->id[i].dim_t == 1 && _tensor_in0->id[i].stride == 1 && _tensor_out->id[i].stride == 1) {  // m dimension
                _tensor_in0->id[i].exec_t = 1;
                _tensor_in1->id[i].exec_t = 1;
                _tensor_out->id[i].exec_t = 1;
                _prim_m_id = i;
                found_m = true;
            } else if (_tensor_in1->id[i].dim_t == 3 && _tensor_in1->id[i].stride == 1) {  // k dimension
                _tensor_in0->id[i].exec_t = 1;
                _tensor_in1->id[i].exec_t = 1;
                _tensor_out->id[i].exec_t = 1;
                _prim_k_id = i;
                found_k = true;
            }
        }

        // set N Primitiv
        for (size_t i = _tensor_in1->id.size() - 1; i >= 0; i--) {
            if (_tensor_in1->id[i].dim_t == 2) {
                _tensor_in0->id[i].exec_t = 1;
                _tensor_in1->id[i].exec_t = 1;
                _tensor_out->id[i].exec_t = 1;
                _prim_n_id = i;
                found_n = true;
                break;
            }
        }

        // set other dimensions to seq
        for (size_t i = 0; i < _tensor_in0->id.size(); i++) {
            if (_tensor_in0->id[i].exec_t != 1) {
                _tensor_in0->id[i].exec_t = 0;
            }
            if (_tensor_in1->id[i].exec_t != 1) {
                _tensor_in1->id[i].exec_t = 0;
            }
            if (_tensor_out->id[i].exec_t != 1) {
                _tensor_out->id[i].exec_t = 0;
            }
        }

        // set loop_ids
        _loop_order.clear();
        for (size_t i = 0; i < _tensor_in0->id.size(); i++) {
            if (_tensor_in0->id[i].exec_t == 0) {
                _loop_order.push_back(i);
            }
        }

        // set first primitive loop id
        _id_first_primitive_loop = _loop_order.size();

        if (found_m && found_k && found_n) {
            return TensorOperation::error_t::success;
        } else {
            std::cerr << "Failed to identify primitives: "
                      << "M found: " << found_m
                      << ", N found: " << found_n
                      << ", K found: " << found_k << std::endl;
            return TensorOperation::error_t::optimize_failed;
        }
    }

    TensorOperation::error_t TensorOperation::compile() {
        _brgemm.generate(_tensor_in0->id[_prim_m_id].dim_sizes,
                         _tensor_in1->id[_prim_n_id].dim_sizes,
                         _tensor_in0->id[_prim_k_id].dim_sizes,
                         1,
                         0,
                         0,
                         0,
                         static_cast<mini_jit::generator::Brgemm::dtype_t>(_dtype));
        _brgemm_kernel = _brgemm.get_kernel();

        if (!_brgemm_kernel) {
            std::cerr << "Failed to compile BRGEMM kernel." << std::endl;
            return TensorOperation::error_t::compile_failed;
        }

        /* TODO get correct lda for bad einsums */
        // define lda, ldb and ldc
        _lda = _tensor_in0->id[_prim_m_id].dim_sizes;
        _ldb = _tensor_in1->id[_prim_k_id].dim_sizes;
        _ldc = _tensor_out->id[_prim_m_id].dim_sizes;

        return TensorOperation::error_t::success;
    }

    void TensorOperation::execute_iter_parallel(const char* ptr_in0,
                                                const char* ptr_in1,
                                                char* ptr_out,
                                                bool first_access,
                                                bool last_access) {
#pragma omp parallel for
        for (int64_t l_it = 0; l_it < _size_parallel_loop; l_it++) {
            // derive if this is first or last access to the output block

            // update pointer with strides
            char* l_ptr_in0 = const_cast<char*>(ptr_in0) + l_it * _tensor_in0->id[_loop_order[0]].stride * 4;
            char* l_ptr_in1 = const_cast<char*>(ptr_in1) + l_it * _tensor_in1->id[_loop_order[0]].stride * 4;
            char* l_ptr_out = ptr_out + l_it * _tensor_out->id[_loop_order[0]].stride * 4;

            if (1 < _id_first_primitive_loop) {
                execute_iter(1,
                             l_ptr_in0,
                             l_ptr_in0,
                             l_ptr_in0,
                             first_access,
                             last_access);
            } else {
                // call first touch kernel if necessary

                // call main kernel
                _brgemm_kernel(l_ptr_in0,
                               l_ptr_in1,
                               l_ptr_out,
                               _lda,
                               _ldb,
                               _ldc,
                               0,
                               0);

                // call last touch kernel if necessary
            }
        }
    }

    void TensorOperation::print() {
    }
}  // namespace einsum::backend
