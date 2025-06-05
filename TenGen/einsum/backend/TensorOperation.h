#ifndef TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H
#define TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H

#include <omp.h>

#include <cstdint>
#include <span>
#include <vector>

#include "TenGen/mini_jit/generator/Brgemm.h"
#include "TenGen/mini_jit/generator/Unary.h"
#include "TenGen/types/Structs.h"
#include "TenGen/types/Types.h"

using namespace TenGen::Types;
using namespace TenGen::Structs;

using namespace TenGen::Types;
using namespace TenGen::Structs;
using Brgemm = TenGen::MiniJit::Generator::Brgemm;
using Unary = TenGen::MiniJit::Generator::Unary;

namespace TenGen::Einsum::Backend {

    class TensorOperation {
       public:
        // scalars
        dtype_t _dtype;
        prim_t _prim_first_touch;
        prim_t _prim_main;
        prim_t _prim_last_touch;
        int64_t _id_first_primitive_loop;

        int64_t _id_prim_m;
        int64_t _id_prim_n;
        int64_t _id_prim_k = 0;
        int64_t _id_prim_br_size = -1;

        int64_t _lda;
        int64_t _ldb;
        int64_t _ldc;
        int64_t _in0_br_stride;
        int64_t _in1_br_stride;

        // owned storage
        std::vector<dim_t> _dim_types_storage;
        std::vector<exec_t> _exec_types_storage;
        std::vector<int64_t> _dim_sizes_storage;
        std::vector<int64_t> _strides_in0_storage;
        std::vector<int64_t> _strides_in1_storage;
        std::vector<int64_t> _strides_out_storage;
        std::vector<int64_t> _loop_sizes_storage;

        // views (spans)
        std::span<const dim_t> _dim_types;
        std::span<const exec_t> _exec_types;
        std::span<const int64_t> _dim_sizes;
        std::span<const int64_t> _strides_in0;
        std::span<const int64_t> _strides_in1;
        std::span<const int64_t> _strides_out;
        std::span<const int64_t> _loop_sizes;

        Brgemm _brgemm;
        Brgemm::kernel_t _brgemm_kernel{nullptr};

        // Unary first touch
        Unary _unary_first_touch;
        Unary::kernel_t _unary_first_touch_kernel{nullptr};

        // Unary last touch
        Unary _unary_last_touch;
        Unary::kernel_t _unary_last_touch_kernel{nullptr};

        // Function to initialize a TensorOperation
        TenGen::Types::error_t setup(dtype_t dtype,
                                     prim_t prim_first_touch,
                                     prim_t prim_main,
                                     prim_t prim_last_touch,
                                     std::vector<dim_t> dim_types,
                                     std::vector<exec_t> exec_types,
                                     std::vector<int64_t> dim_sizes,
                                     std::vector<int64_t> strides_in0,
                                     std::vector<int64_t> strides_in1,
                                     std::vector<int64_t> strides_out) {
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

            // extract the sizes of the sequential loops
            // till we reach the first primitive loop
            for (size_t i = 0; i < _dim_types.size(); i++) {
                // if the execution type is not a primitive,
                // we add the size to the loop sizes storage
                if (_exec_types[i] != exec_t::prim) {
                    _loop_sizes_storage.push_back(_dim_sizes[i]);
                    // otherwise, we set the id of the first primitive loop
                    // and break the loop
                } else {
                    _id_first_primitive_loop = i;
                    break;
                }
            }

            // remap the loop sizes to a span
            _loop_sizes = _loop_sizes_storage;

            // again, go through the dimensions and now only
            // do something if the execution type is a primitive
            for (size_t i = 0; i < _dim_sizes.size(); i++) {
                // check if the dimension is a primitive and if it is the m loop
                if (_dim_types[i] == dim_t::m && _exec_types[i] == exec_t::prim) {
                    _id_prim_m = i;
                    // check if the dimension is a primitive and if it is the n loop
                } else if (_dim_types[i] == dim_t::n && _exec_types[i] == exec_t::prim) {
                    _id_prim_n = i;
                    // check if the dimension is a primitive and if it is the k loop
                } else if (_dim_types[i] == dim_t::k && _exec_types[i] == exec_t::prim) {
                    // if we have not set the id of the k loop yet, we set it
                    if (_id_prim_k == 0) {
                        _id_prim_k = i;
                        // if we set it already and encounter a new k loop
                        // we know that we have a batch-reduced size
                    } else {
                        _id_prim_br_size = _id_prim_k;
                        _id_prim_k = i;
                    }
                }
            }

            // create brgemm_kernel form that primitives above
            _brgemm.generate(_dim_sizes[_id_prim_m],
                             _dim_sizes[_id_prim_n],
                             _dim_sizes[_id_prim_k],
                             (_id_prim_br_size > -1) ? _dim_sizes[_id_prim_br_size] : 1,  // batch-reduce size or gemm if no br size
                             0,
                             0,
                             0,
                             static_cast<dtype_t>(_dtype));
            _brgemm_kernel = _brgemm.get_kernel();

            // check if we have a first touch primitive
            // for now this only applys to zero
            if (_prim_first_touch != prim_t::none) {
                _unary_first_touch.generate(_dim_sizes[_id_prim_m],
                                            _dim_sizes[_id_prim_m],
                                            0,
                                            dtype_t::fp32,
                                            ptype_t::zero);
            }
            _unary_first_touch_kernel = _unary_first_touch.get_kernel();

            // check if we have a last touch primitive
            // for now this only applys to relu
            if (_prim_last_touch != prim_t::none) {
                _unary_last_touch.generate(_dim_sizes[_id_prim_m],
                                           _dim_sizes[_id_prim_m],
                                           0,
                                           dtype_t::fp32,
                                           ptype_t::relu);
            }
            _unary_last_touch_kernel = _unary_last_touch.get_kernel();

            // set lda, ldb, ldc, in0_br_stride, in1_br_stride
            // TODO: currently assumes primitve types are always the last 3 dimensions
            _lda = _strides_in0[_strides_in0.size() - 1];
            _ldb = _strides_in1[_strides_in1.size() - 2];
            _ldc = _strides_out[_strides_out.size() - 2];

            _in0_br_stride = _strides_in0[_strides_in0.size() - 4];
            _in1_br_stride = _strides_in1[_strides_in1.size() - 4];

// this is really cool
#ifdef DEBUG
            // print all necessary information
            std::cout << "TensorOperation setup:" << std::endl;
            std::cout << "  dtype: " << static_cast<int>(_dtype) << std::endl;
            std::cout << "  prim_first_touch: " << static_cast<int>(_prim_first_touch) << std::endl;
            std::cout << "  prim_main: " << static_cast<int>(_prim_main) << std::endl;
            std::cout << "  prim_last_touch: " << static_cast<int>(_prim_last_touch) << std::endl;
            std::cout << "  id_first_primitive_loop: " << _id_first_primitive_loop << std::endl;
            std::cout << "  id_prim_m: " << _id_prim_m << std::endl;
            std::cout << "  id_prim_n: " << _id_prim_n << std::endl;
            std::cout << "  id_prim_k: " << _id_prim_k << std::endl;
            std::cout << "  id_prim_br_size: " << _id_prim_br_size << std::endl;
            std::cout << "  loop_sizes: ";
            for (const auto& size : _loop_sizes) {
                std::cout << size << " ";
            }
            std::cout << std::endl;
            std::cout << "M: " << _dim_sizes[_id_prim_m] << std::endl;
            std::cout << "N: " << _dim_sizes[_id_prim_n] << std::endl;
            std::cout << "K: " << _dim_sizes[_id_prim_k] << std::endl;
            std::cout << "BR size: " << ((_id_prim_br_size > -1) ? _dim_sizes[_id_prim_br_size] : 1) << std::endl;
            std::cout << "lda: " << _lda << std::endl;
            std::cout << "ldb: " << _ldb << std::endl;
            std::cout << "ldc: " << _ldc << std::endl;
            std::cout << "in0_br_stride: " << _in0_br_stride << std::endl;
            std::cout << "in1_br_stride: " << _in1_br_stride << std::endl;

            std::cout << "***********************" << std::endl;
#endif

            return TenGen::Types::error_t::success;
        }

        // Function to execute a tensor operation
        void execute(const void* tensor_in0,
                     const void* tensor_in1,
                     void* tensor_out) {
            // get pointers to input and output data
            char const* l_ptr_in0 = static_cast<char const*>(tensor_in0);
            char const* l_ptr_in1 = static_cast<char const*>(tensor_in1);
            char* l_ptr_out = static_cast<char*>(tensor_out);

            // execute the operation
            execute_iter(0, l_ptr_in0, l_ptr_in1, l_ptr_out, false, false);
        }

        // Recursive loop execution function
        void execute_iter(int64_t id_loop,
                          const char* ptr_in0,
                          const char* ptr_in1,
                          char* ptr_out,
                          bool first_access,
                          bool last_access) {
            // go through each sequential loop (M, N, K) recursively
            int64_t l_size = _loop_sizes[id_loop];
            // apply the loop
            for (int64_t l_it = 0; l_it < l_size; l_it++) {
                // calculate the pointers for the current iteration
                char* l_ptr_in0 = const_cast<char*>(ptr_in0) + l_it * _strides_in0[id_loop] * 4;
                char* l_ptr_in1 = const_cast<char*>(ptr_in1) + l_it * _strides_in1[id_loop] * 4;
                char* l_ptr_out = ptr_out + l_it * _strides_out[id_loop] * 4;

                // TODO: handle first and last access
                // if alle squential loops are applied, we can execute the primitive
                if (id_loop + 1 < _id_first_primitive_loop) {
                    execute_iter(id_loop + 1,
                                 l_ptr_in0,
                                 l_ptr_in1,
                                 l_ptr_out,
                                 first_access,
                                 last_access);
                } else {
                    // handle first touch
                    if (first_access && _prim_first_touch != prim_t::none) {
                        // TODO
                        _unary_first_touch_kernel(l_ptr_in0, l_ptr_out, _ldc, _ldc);
                    }
                    // do the brgemm operation
                    _brgemm_kernel(l_ptr_in0, l_ptr_in1, l_ptr_out,
                                   _lda,
                                   _ldb,
                                   _ldc,
                                   _in0_br_stride,
                                   _in1_br_stride);

                    // handle last touch
                    if (last_access && _prim_last_touch != prim_t::none) {
                        // TODO
                        _unary_last_touch_kernel(l_ptr_out, l_ptr_out, _ldc, _ldc);
                    }
                }
            }
        }

        /**
         * General-purpose loop implementation featuring first and last touch operations with OMP parallelization.
         *
         * @param ptr_in0      Pointer to the first input tensor's data.
         * @param ptr_in1      Pointer to the second input tensor's data (use nullptr if unary).
         * @param ptr_out      Pointer to the output tensor's data.
         * @param first_access True if first time accessing data of output tensor.
         * @param last_access  True if last time accessing data of output tensor.
         **/
        void execute_iter_parallel(const void* ptr_in0,
                                   const void* ptr_in1,
                                   void* ptr_out,
                                   bool first_access,
                                   bool last_access) {
            int64_t num_parallel_loops = 0;
            int64_t size_parallel_loops = 1;
            for (exec_t dim : _exec_types) {
                if (dim == exec_t::shared) {
                    size_parallel_loops *= _loop_sizes[num_parallel_loops];
                    num_parallel_loops++;
                }
            }
#pragma omp parallel for
            for (int64_t it_all = 0; it_all < size_parallel_loops; it_all++) {
                int64_t it_remaining = it_all;

                bool is_first = (it_all == 0);
                bool is_last = (it_all == size_parallel_loops - 1);

                const char* temp_ptr_in0 = static_cast<const char*>(ptr_in0);
                const char* temp_ptr_in1 = static_cast<const char*>(ptr_in1);
                char* temp_ptr_out = static_cast<char*>(ptr_out);

                for (int64_t id_loop = num_parallel_loops - 1; id_loop >= 0; id_loop--) {
                    // calculate loop index l_it for loop l_id_loop
                    int64_t it = it_remaining % _loop_sizes[id_loop];
                    it_remaining = it_remaining / _loop_sizes[id_loop];

                    std::cout << "it_remaining: " << it_remaining << ", it: " << it << ", loop_size: " << _loop_sizes[id_loop] << ", it_all: " << it_all << ", id_loop: " << id_loop << std::endl;

                    // update pointer with strides
                    temp_ptr_in0 += it * _strides_in0[id_loop];
                    temp_ptr_in1 += it * _strides_in1[id_loop];
                    temp_ptr_out += it * _strides_out[id_loop];
                }
                // call non parallel loops or kernel

                bool thread_first_access = first_access && (it_all == 0);
                bool thread_last_access = last_access && (it_all == size_parallel_loops - 1);

                execute_iter(num_parallel_loops,
                             temp_ptr_in0,
                             temp_ptr_in1,
                             temp_ptr_out,
                             thread_first_access,
                             thread_last_access);
            }
        }
    };
}  // namespace TenGen::Einsum::Backend
#endif  // TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H
