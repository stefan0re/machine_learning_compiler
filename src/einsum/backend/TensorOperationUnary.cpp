#include "TensorOperationUnary.h"

namespace einsum::backend {
    TensorOperationUnary::error_t TensorOperationUnary::setup(dtype_t dtype,
                                                              prim_t prim_main,
                                                              std::span<const exec_t> exec_types,
                                                              std::span<const int64_t> dim_sizes,
                                                              std::span<const int64_t> strides_in0,
                                                              std::span<const int64_t> strides_out) {
        // set primitive types and dtype
        _prim_main = prim_main;
        _dtype = dtype;

        // set vectors
        _exec_types.assign(exec_types.begin(), exec_types.end());
        _dim_sizes.assign(dim_sizes.begin(), dim_sizes.end());
        _strides_in0.assign(strides_in0.begin(), strides_in0.end());
        _strides_out.assign(strides_out.begin(), strides_out.end());

        return TensorOperationUnary::error_t::success;
    }

    TensorOperationUnary::error_t TensorOperationUnary::compile() {
        if (_prim_main == prim_t::trans) {
            for (size_t i = 0; i < _exec_types.size(); i++) {
                _loop_ids.push_back(i);
            }
            return TensorOperationUnary::error_t::success;
        }

        // check for loop dimensions
        for (size_t i = 0; i < _exec_types.size(); i++) {
            if (_exec_types[i] != exec_t::prim) {
                _loop_ids.push_back(i);
            }
        }

        if (_prim_main != prim_t::trans) {
            for (size_t i = 0; i < _exec_types.size(); i++) {
                if (_exec_types[i] == exec_t::prim && _strides_in0[i] == 1 && _strides_out[i] == 1) {
                    _id_prim_m = i;
                } else {
                    _id_prim_n = i;
                }
            }
        } else {
            for (size_t i = 0; i < _exec_types.size(); i++) {
                if (_exec_types[i] == exec_t::prim && _strides_in0[i] == 1) {
                    _id_prim_m = i;
                } else if (_exec_types[i] == exec_t::prim && _strides_out[i] == 1) {
                    _id_prim_n = i;
                }
            }
        }

        for (const auto& id : _loop_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        _unary.generate(_dim_sizes[_id_prim_m],
                        _dim_sizes[_id_prim_n],
                        static_cast<mini_jit::generator::Unary::dtype_t>(_dtype),
                        static_cast<mini_jit::generator::Unary::ptype_t>(_prim_main));

        _unary_kernel = _unary.get_kernel();

        // leading dimensions
        _ldi = _strides_in0[_id_prim_n];
        if (_prim_main != prim_t::trans) {
            _ldo = _strides_out[_id_prim_n];
        } else {
            _ldo = _strides_out[_id_prim_m];
        }

        return TensorOperationUnary::error_t::success;
    }

    void TensorOperationUnary::execute(void const* tensor_in,
                                       void* tensor_out) {
        // get pointers to input and output data
        const char* l_ptr_in = static_cast<const char*>(tensor_in);
        char* l_ptr_out = static_cast<char*>(tensor_out);

        // execute the unary operation
        execute_iter(0, l_ptr_in, l_ptr_out);
    }

    void TensorOperationUnary::execute_iter(int64_t id_loop,
                                            const char* ptr_in,
                                            char* ptr_out) {
        int64_t l_size = 1;
        if (_loop_ids.size() > 0) {
            l_size = _dim_sizes[_loop_ids[id_loop]];
        }

        for (int64_t l_it = 0; l_it < l_size; l_it++) {
            // calculate the input and output index
            char* l_ptr_in0 = const_cast<char*>(ptr_in);
            char* l_ptr_out = ptr_out;

            if (_loop_ids.size() > 0) {
                l_ptr_in0 += l_it * _strides_in0[_loop_ids[id_loop]] * 4;
                l_ptr_out += l_it * _strides_out[_loop_ids[id_loop]] * 4;
            }

            if (_prim_main == prim_t::trans) {
                if ((_loop_ids.size() > 0) && (id_loop < _loop_ids.size() - 1)) {
                    // recursive function call
                    execute_iter(id_loop + 1,
                                 l_ptr_in0,
                                 l_ptr_out);

                } else {
                    float* input_pointer = reinterpret_cast<float*>(const_cast<char*>(l_ptr_in0));
                    float* output_pointer = reinterpret_cast<float*>(l_ptr_out);
                    output_pointer[0] = input_pointer[0];
                }
            } else {
                if ((_loop_ids.size() > 0) && (id_loop < _loop_ids.size() - 1)) {
                    // recursive function call
                    execute_iter(id_loop + 1,
                                 l_ptr_in0,
                                 l_ptr_out);
                } else {
                    // call main kernel
                    _unary_kernel(l_ptr_in0, l_ptr_out, _ldi, _ldo);
                }
            }
        }
    }
}  // namespace einsum::backend
