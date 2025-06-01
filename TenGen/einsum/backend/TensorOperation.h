#ifndef TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H
#define TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H

#include <cstdint>
#include <span>
#include <vector>

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

            // safely copies all input arrays so they outlive setup()
            // 1. copy data into owned storage
            op._dim_types_storage.assign(dim_types.begin(), dim_types.end());
            op._exec_types_storage.assign(exec_types.begin(), exec_types.end());
            op._dim_sizes_storage.assign(dim_sizes.begin(), dim_sizes.end());
            op._strides_in0_storage.assign(strides_in0.begin(), strides_in0.end());
            op._strides_in1_storage.assign(strides_in1.begin(), strides_in1.end());
            op._strides_out_storage.assign(strides_out.begin(), strides_out.end());

            // 2. set spans to refer to owned data
            op.dim_types = op._dim_types_storage;
            op.exec_types = op._exec_types_storage;
            op.dim_sizes = op._dim_sizes_storage;
            op.strides_in0 = op._strides_in0_storage;
            op.strides_in1 = op._strides_in1_storage;
            op.strides_out = op._strides_out_storage;

            // extract the sizes of the sequential loops
            // till we reach the first primitive loop
            for (size_t i = 0; i < op.dim_types.size(); i++) {
                // if the execution type is not a primitive,
                // we add the size to the loop sizes storage
                if (op.exec_types[i] != exec_t::prim) {
                    op._loop_sizes_storage.push_back(op.dim_sizes[i]);
                    // otherwise, we set the id of the first primitive loop
                    // and break the loop
                } else {
                    op.id_first_primitive_loop = i;
                    break;
                }
            }

            // remap the loop sizes to a span
            op.loop_sizes = op._loop_sizes_storage;

            // again, go through the dimensions and now only
            // do something if the execution type is a primitive
            for (size_t i = 0; i < op.dim_sizes.size(); i++) {
                // check if the dimension is a primitive and if it is the m loop
                if (op.dim_types[i] == dim_t::m && op.exec_types[i] == exec_t::prim) {
                    op.id_prim_m = i;
                    // check if the dimension is a primitive and if it is the n loop
                } else if (op.dim_types[i] == dim_t::n && op.exec_types[i] == exec_t::prim) {
                    op.id_prim_n = i;
                    // check if the dimension is a primitive and if it is the k loop
                } else if (op.dim_types[i] == dim_t::k && op.exec_types[i] == exec_t::prim) {
                    // if we have not set the id of the k loop yet, we set it
                    if (op.id_prim_k == 0) {
                        op.id_prim_k = i;
                        // if we set it already and encounter a new k loop
                        // we know that we have a batch-reduced size
                    } else {
                        op.id_prim_br_size = op.id_prim_k;
                        op.id_prim_k = i;
                    }
                }
            }

            // create brgemm_kernel form that primitives above
            op.brgemm.generate(op.dim_sizes[op.id_prim_m],
                               op.dim_sizes[op.id_prim_n],
                               op.dim_sizes[op.id_prim_k],
                               (op.id_prim_br_size > -1) ? op.dim_sizes[op.id_prim_br_size] : 1,  // batch-reduce size or gemm if no br size
                               0,
                               0,
                               0,
                               static_cast<dtype_t>(op.dtype));
            op.brgemm_kernel = op.brgemm.get_kernel();

            // check if we have a first touch primitive
            // for now this only applys to zero
            if (op.prim_first_touch != prim_t::none) {
                op.unary_first_touch.generate(op.dim_sizes[op.id_prim_m],
                                              op.dim_sizes[op.id_prim_m],
                                              0,
                                              dtype_t::fp32,
                                              ptype_t::zero);
            }
            op.unary_first_touch_kernel = op.unary_first_touch.get_kernel();

            // check if we have a last touch primitive
            // for now this only applys to relu
            if (op.prim_last_touch != prim_t::none) {
                op.unary_last_touch.generate(op.dim_sizes[op.id_prim_m],
                                             op.dim_sizes[op.id_prim_m],
                                             0,
                                             dtype_t::fp32,
                                             ptype_t::relu);
            }
            op.unary_last_touch_kernel = op.unary_last_touch.get_kernel();

            // set lda, ldb, ldc, in0_br_stride, in1_br_stride
            // TODO: currently assumes primitve types are always the last 3 dimensions
            op.lda = op.strides_in0[op.strides_in0.size() - 1];
            op.ldb = op.strides_in1[op.strides_in1.size() - 2];
            op.ldc = op.strides_out[op.strides_out.size() - 2];

            op.in0_br_stride = op.strides_in0[op.strides_in0.size() - 4];
            op.in1_br_stride = op.strides_in1[op.strides_in1.size() - 4];

// this is really cool
#ifdef DEBUG
            // print all necessary information
            std::cout << "TensorOperation setup:" << std::endl;
            std::cout << "  dtype: " << static_cast<int>(op.dtype) << std::endl;
            std::cout << "  prim_first_touch: " << static_cast<int>(op.prim_first_touch) << std::endl;
            std::cout << "  prim_main: " << static_cast<int>(op.prim_main) << std::endl;
            std::cout << "  prim_last_touch: " << static_cast<int>(op.prim_last_touch) << std::endl;
            std::cout << "  id_first_primitive_loop: " << op.id_first_primitive_loop << std::endl;
            std::cout << "  id_prim_m: " << op.id_prim_m << std::endl;
            std::cout << "  id_prim_n: " << op.id_prim_n << std::endl;
            std::cout << "  id_prim_k: " << op.id_prim_k << std::endl;
            std::cout << "  id_prim_br_size: " << op.id_prim_br_size << std::endl;
            std::cout << "  loop_sizes: ";
            for (const auto& size : op.loop_sizes) {
                std::cout << size << " ";
            }
            std::cout << std::endl;
            std::cout << "M: " << op.dim_sizes[op.id_prim_m] << std::endl;
            std::cout << "N: " << op.dim_sizes[op.id_prim_n] << std::endl;
            std::cout << "K: " << op.dim_sizes[op.id_prim_k] << std::endl;
            std::cout << "BR size: " << ((op.id_prim_br_size > -1) ? op.dim_sizes[op.id_prim_br_size] : 1) << std::endl;
            std::cout << "lda: " << op.lda << std::endl;
            std::cout << "ldb: " << op.ldb << std::endl;
            std::cout << "ldc: " << op.ldc << std::endl;
            std::cout << "in0_br_stride: " << op.in0_br_stride << std::endl;
            std::cout << "in1_br_stride: " << op.in1_br_stride << std::endl;

            std::cout << "***********************" << std::endl;
#endif

            return error_t::success;
        }

        // Function to execute a tensor operation
        void execute(const TensorConfig& op,
                     const void* tensor_in0,
                     const void* tensor_in1,
                     void* tensor_out) {
            // get pointers to input and output data
            char const* l_ptr_in0 = static_cast<char const*>(tensor_in0);
            char const* l_ptr_in1 = static_cast<char const*>(tensor_in1);
            char* l_ptr_out = static_cast<char*>(tensor_out);

            // execute the operation
            execute_iter(op, 0, l_ptr_in0, l_ptr_in1, l_ptr_out, false, false);
        }

        // Recursive loop execution function
        void execute_iter(const TensorConfig& op,
                          int64_t id_loop,
                          const char* ptr_in0,
                          const char* ptr_in1,
                          char* ptr_out,
                          bool first_access,
                          bool last_access) {
            // go through each sequential loop (M, N, K) recursively
            int64_t l_size = op.loop_sizes[id_loop];
            // apply the loop
            for (int64_t l_it = 0; l_it < l_size; l_it++) {
                // calculate the pointers for the current iteration
                char* l_ptr_in0 = const_cast<char*>(ptr_in0) + l_it * op.strides_in0[id_loop] * 4;
                char* l_ptr_in1 = const_cast<char*>(ptr_in1) + l_it * op.strides_in1[id_loop] * 4;
                char* l_ptr_out = ptr_out + l_it * op.strides_out[id_loop] * 4;

                // TODO: handle first and last access
                // if alle squential loops are applied, we can execute the primitive
                if (id_loop + 1 < op.id_first_primitive_loop) {
                    execute_iter(op,
                                 id_loop + 1,
                                 l_ptr_in0,
                                 l_ptr_in1,
                                 l_ptr_out,
                                 first_access,
                                 last_access);
                } else {
                    // handle first touch
                    if (first_access && op.prim_first_touch != prim_t::none) {
                        // TODO
                        op.unary_first_touch_kernel(l_ptr_in0, l_ptr_out, op.ldc, op.ldc);
                    }
                    // do the brgemm operation
                    op.brgemm_kernel(l_ptr_in0, l_ptr_in1, l_ptr_out,
                                     op.lda,
                                     op.ldb,
                                     op.ldc,
                                     op.in0_br_stride,
                                     op.in1_br_stride);

                    // handle last touch
                    if (last_access && op.prim_last_touch != prim_t::none) {
                        // TODO
                        op.unary_last_touch_kernel(l_ptr_out, l_ptr_out, op.ldc, op.ldc);
                    }
                }
            }
        }
    };
}  // namespace TenGen::Einsum::Backend
#endif  // TENGEN_EINSUM_BACKEND_TENSOR_OPERATION_H
