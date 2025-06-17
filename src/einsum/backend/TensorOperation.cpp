#include "TensorOperation.h"

#include <iostream>

#include "../../mini_jit/generator/Brgemm.h"
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
                                       bool last_access,
                                       int64_t loop_count) {
        int64_t l_size = _loop_sizes[id_loop];
        loop_count++;

        for (int64_t l_it = 0; l_it < l_size; l_it++) {
            char* l_ptr_in0 = const_cast<char*>(ptr_in0) + l_it * _strides_in0[id_loop] * 4;
            char* l_ptr_in1 = const_cast<char*>(ptr_in1) + l_it * _strides_in1[id_loop] * 4;
            char* l_ptr_out = ptr_out + l_it * _strides_out[id_loop] * 4;

            if (loop_count < _loop_order.size()) {
                execute_iter(_loop_order[loop_count],
                             l_ptr_in0,
                             l_ptr_in1,
                             l_ptr_out,
                             first_access,
                             last_access,
                             loop_count);

            } else {
                // handle first touch
                _brgemm_kernel(l_ptr_in0, l_ptr_in1, l_ptr_out,
                               _lda,
                               _ldb,
                               _ldc,
                               _in0_br_stride,
                               _in1_br_stride);

                // TODO: handle last touch
            }
        }
    }

    TensorOperation::error_t TensorOperation::optimize() {
        fuse_dimensions();
        split_dimensions();
        identify_primitives();
        reorder_dimensions();

        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::split_dimensions() {
        // get M loop IDs
        std::vector<int64_t> m_loop_ids;
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::m) {
                m_loop_ids.push_back(i);
            }
        }

        // Split M dimension if it is bigger that 64
        for (size_t i = 0; i < m_loop_ids.size(); i++) {
            if (_dim_sizes[m_loop_ids[i]] > 64) {
                std::vector<int64_t> prims = prime_factors(_dim_sizes[m_loop_ids[i]]);

                // select prime factor smaller than 64
                int64_t new_size = find_new_size(prims);

                int64_t other_size = _dim_sizes[m_loop_ids[i]] / new_size;

                // insert dimension type
                _dim_types_storage.insert(_dim_types_storage.begin() + m_loop_ids[i] + 1, dim_t::m);
                _dim_types = _dim_types_storage;

                // insert dimension size
                _dim_sizes_storage.insert(_dim_sizes_storage.begin() + m_loop_ids[i], other_size);
                _dim_sizes_storage.insert(_dim_sizes_storage.begin() + m_loop_ids[i] + 1, new_size);
                _dim_sizes_storage.erase(_dim_sizes_storage.begin() + m_loop_ids[i] + 2);
                _dim_sizes = _dim_sizes_storage;

                // insert stride in0
                _strides_in0_storage.insert(_strides_in0_storage.begin() + m_loop_ids[i], new_size * _strides_in0[m_loop_ids[i]]);
                _strides_in0 = _strides_in0_storage;

                // insert stride in1
                _strides_in1_storage.insert(_strides_in1_storage.begin() + m_loop_ids[i] + 1, 0);
                _strides_in1 = _strides_in1_storage;

                // insert stride out
                _strides_out_storage.insert(_strides_out_storage.begin() + m_loop_ids[i], new_size * _strides_out[m_loop_ids[i]]);
                _strides_out = _strides_out_storage;
            }
        }

        // get K loop IDs
        std::vector<int64_t> k_loop_ids;
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::k) {
                k_loop_ids.push_back(i);
            }
        }
        // Split K dimension if it is bigger that 128
        for (size_t i = 0; i < k_loop_ids.size(); i++) {
            if (_dim_sizes[k_loop_ids[i]] > 128) {
                std::vector<int64_t> prims = prime_factors(_dim_sizes[k_loop_ids[i]]);

                // select prime factor smaller than 128
                int64_t new_size = find_new_size(prims);

                int64_t other_size = _dim_sizes[k_loop_ids[i]] / new_size;

                // insert dimension type
                _dim_types_storage.insert(_dim_types_storage.begin() + k_loop_ids[i] + 1, dim_t::k);
                _dim_types = _dim_types_storage;

                // insert dimension size
                _dim_sizes_storage.insert(_dim_sizes_storage.begin() + k_loop_ids[i], new_size);
                _dim_sizes_storage.insert(_dim_sizes_storage.begin() + k_loop_ids[i] + 1, other_size);
                _dim_sizes_storage.erase(_dim_sizes_storage.begin() + k_loop_ids[i] + 2);
                _dim_sizes = _dim_sizes_storage;

                // insert stride in0
                _strides_in0_storage.insert(_strides_in0_storage.begin() + k_loop_ids[i], new_size * _strides_in0[k_loop_ids[i]]);
                _strides_in0 = _strides_in0_storage;

                // insert stride in1
                _strides_in1_storage.insert(_strides_in1_storage.begin() + k_loop_ids[i] + 1, new_size * _strides_in1[k_loop_ids[i]]);
                _strides_in1 = _strides_in1_storage;

                // insert stride out
                _strides_out_storage.insert(_strides_out_storage.begin() + k_loop_ids[i], new_size * _strides_out[k_loop_ids[i]]);
                _strides_out = _strides_out_storage;
            }
        }

        // get N loop IDs
        std::vector<int64_t> n_loop_ids;
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::n) {
                n_loop_ids.push_back(i);
            }
        }

        // Split N dimension if it is bigger that 64
        for (size_t i = 0; i < n_loop_ids.size(); i++) {
            if (_dim_sizes[n_loop_ids[i]] > 64) {
                std::vector<int64_t> prims = prime_factors(_dim_sizes[n_loop_ids[i]]);

                // select prime factor smaller than 64
                int64_t new_size = find_new_size(prims);

                int64_t other_size = _dim_sizes[n_loop_ids[i]] / new_size;

                // insert dimension type
                _dim_types_storage.insert(_dim_types_storage.begin() + n_loop_ids[i] + 1, dim_t::n);
                _dim_types = _dim_types_storage;

                // insert dimension size
                _dim_sizes_storage.insert(_dim_sizes_storage.begin() + n_loop_ids[i], new_size);
                _dim_sizes_storage.insert(_dim_sizes_storage.begin() + n_loop_ids[i] + 1, other_size);
                _dim_sizes_storage.erase(_dim_sizes_storage.begin() + n_loop_ids[i] + 2);
                _dim_sizes = _dim_sizes_storage;

                // insert stride in0
                _strides_in0_storage.insert(_strides_in0_storage.begin() + n_loop_ids[i], 0);
                _strides_in0 = _strides_in0_storage;

                // insert stride in1
                _strides_in1_storage.insert(_strides_in1_storage.begin() + n_loop_ids[i] + 1, new_size * _strides_in1[n_loop_ids[i]]);
                _strides_in1 = _strides_in1_storage;

                // insert stride out
                _strides_out_storage.insert(_strides_out_storage.begin() + n_loop_ids[i], new_size * _strides_out[n_loop_ids[i]]);
                _strides_out = _strides_out_storage;
            }
        }

        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::fuse_dimensions() {
        // get M loop IDs
        std::vector<int64_t> m_loop_ids;
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::m) {
                m_loop_ids.push_back(i);
            }
        }

        for (size_t i = 0; i < m_loop_ids.size(); i++) {
            if (_dim_sizes[m_loop_ids[i]] < 16) {
                if (i + 1 < m_loop_ids.size() && _dim_types[m_loop_ids[i + 1]] == dim_t::m) {
                    _dim_sizes_storage[m_loop_ids[i]] *= _dim_sizes[m_loop_ids[i + 1]];

                    _dim_types_storage.erase(_dim_types_storage.begin() + m_loop_ids[i + 1]);
                    _dim_sizes_storage.erase(_dim_sizes_storage.begin() + m_loop_ids[i + 1]);

                    _strides_in0_storage[m_loop_ids[i]] *= _dim_sizes[m_loop_ids[i + 1]];
                    _strides_in0_storage.erase(_strides_in0_storage.begin() + m_loop_ids[i + 1]);

                    _strides_in1_storage.erase(_strides_in1_storage.begin() + m_loop_ids[i + 1]);

                    _strides_out_storage[m_loop_ids[i]] *= _dim_sizes[m_loop_ids[i + 1]];
                    _strides_out_storage.erase(_strides_out_storage.begin() + m_loop_ids[i + 1]);
                }
            }
        }

        // TODO: better fusing
        // TOOD: fuse other dim types like N and K

        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::reorder_dimensions() {
        // count number of loop dimensions
        int64_t num_loops = 0;
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_exec_types[i] == exec_t::seq) {
                num_loops++;
            }
        }

        if (num_loops < 1) {
            return TensorOperation::error_t::success;
        }

        _loop_order_storage.clear();
        _loop_order_storage.resize(num_loops);

        // put seq id loops inside loop order structure
        int64_t loop_id = 0;
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_exec_types[i] == exec_t::seq) {
                _loop_order_storage[loop_id] = i;
                loop_id++;
            }
        }

        // interleave M and N loops
        for (size_t i = 0; i < _loop_order_storage.size(); i++) {
            if (_dim_types[_loop_order_storage[i]] == dim_t::m) {
                for (size_t j = i + 1; j < _loop_order_storage.size(); j++) {
                    if (_dim_types[_loop_order_storage[j]] == dim_t::n) {
                        std::swap(_loop_order_storage[i], _loop_order_storage[j]);
                        break;
                    }
                }
            }
        }

        _loop_order = _loop_order_storage;

        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::identify_primitives() {
        // set all dimensions to seq type
        _exec_types_storage.clear();
        _exec_types_storage.resize(_dim_types.size(), exec_t::seq);

        // find stride 1 M dimension left input tensor
        _id_prim_m = -1;
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::m && _strides_in0[i] == 1 && _strides_out[i] == 1) {
                _id_prim_m = i;
                _exec_types_storage[i] = exec_t::prim;
                break;
            }
        }
        _id_prim_k = -1;
        // find stride 1 K dimension in right input tensor
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::k && _strides_in1[i] == 1) {
                _id_prim_k = i;
                _exec_types_storage[i] = exec_t::prim;
                break;
            }
        }

        // find N dimension with lowest stride in right input tensor
        _id_prim_n = -1;
        int64_t min_stride = std::numeric_limits<int64_t>::max();
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::n && _strides_in1[i] < min_stride) {
                min_stride = _strides_in1[i];
                _id_prim_n = i;
            }
        }
        _exec_types_storage[_id_prim_n] = exec_t::prim;

        // find BR dimension be aware of found K dimension with lovest stride
        _id_prim_br = -1;
        min_stride = std::numeric_limits<int64_t>::max();
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::k && _strides_in1[i] < min_stride && i != _id_prim_k) {
                min_stride = _strides_in1[i];
                _id_prim_br = i;
            }
        }
        if (_id_prim_br != -1) {
            _exec_types_storage[_id_prim_br] = exec_t::prim;
            _id_prim_br_size = _dim_sizes[_id_prim_br];
        }

        if (_id_prim_m == -1) {
            std::cerr << "Error: No stride 1 M dimension found in left input tensor." << std::endl;
            return TensorOperation::error_t::optimize_failed;
        } else if (_id_prim_k == -1) {
            std::cerr << "Error: No stride 1 K dimension found in right input tensor." << std::endl;
            return TensorOperation::error_t::optimize_failed;
        } else if (_id_prim_n == -1) {
            std::cerr << "Error: No stride N dimension found in right input tensor." << std::endl;
            return TensorOperation::error_t::optimize_failed;
        }

        // set loop sizes array
        _loop_sizes_storage.clear();
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_exec_types_storage[i] == exec_t::seq) {
                _loop_sizes_storage.push_back(_dim_sizes[i]);
                if (_dim_types[i] != dim_t::k) {
                    _exec_types_storage[i] = exec_t::shared;
                }
            } else {
                _loop_sizes_storage.push_back(1);
            }
        }

        // TODO: identfy parallel dimensions

        _loop_sizes = _loop_sizes_storage;
        _exec_types = _exec_types_storage;

        return TensorOperation::error_t::success;
    }

    TensorOperation::error_t TensorOperation::compile() {
        // create brgemm_kernel
        _brgemm.generate(_dim_sizes[_id_prim_m],
                         _dim_sizes[_id_prim_n],
                         _dim_sizes[_id_prim_k],
                         (_id_prim_br > -1) ? _dim_sizes[_id_prim_br] : 1,  // todo: be carefull currently there is no size!! just an ID
                         0,
                         0,
                         0,
                         static_cast<mini_jit::generator::Brgemm::dtype_t>(_dtype));
        _brgemm_kernel = _brgemm.get_kernel();

        // check if we have a first touch primitive
        // for now this only applys to zero
        // using the size of the output tensor as value for m and n
        if (_prim_first_touch != prim_t::none) {
            _unary_first_touch.generate(_dim_sizes[_id_prim_m],
                                        _dim_sizes[_id_prim_n],
                                        0,
                                        mini_jit::generator::Unary::dtype_t::fp32,

                                        mini_jit::generator::Unary::ptype_t::zero);
            _unary_first_touch_kernel = _unary_first_touch.get_kernel();
        }

        // check if we have a last touch primitive
        // for now this only applys to relu
        if (_prim_last_touch != prim_t::none) {
            _unary_last_touch.generate(_dim_sizes[_id_prim_m],
                                       _dim_sizes[_id_prim_n],
                                       0,
                                       mini_jit::generator::Unary::dtype_t::fp32,
                                       mini_jit::generator::Unary::ptype_t::relu);
            _unary_last_touch_kernel = _unary_last_touch.get_kernel();
        }
        // check if other prim M dimension have the stride of prim M dimension size in the left input tensor
        _lda = _dim_sizes[_id_prim_m];
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::m && _strides_in0[i] == _lda) {
                _lda *= _dim_sizes[i];
            }
        }
        // check if other prim K dimension have the stride of prim K dimension size right input tensor
        _ldb = _dim_sizes[_id_prim_k];
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::k && _strides_in1[i] == _ldb) {
                _ldb *= _dim_sizes[i];
            }
        }

        // check if other prim M dimension have the stride of prim M dimension size in the output tensor
        _ldc = _dim_sizes[_id_prim_m];
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::m && _strides_out[i] == _ldc) {
                _ldc *= _dim_sizes[i];
            }
        }

        _in0_br_stride = _strides_in0[_id_prim_br];
        _in1_br_stride = _strides_in1[_id_prim_br];

        return TensorOperation::error_t::success;
    }

    void TensorOperation::execute_iter_parallel(const char* ptr_in0,
                                                const char* ptr_in1,
                                                char* ptr_out,
                                                bool first_access,
                                                bool last_access) {
        int64_t size_parallel_loops = 1;
        std::cout << "_num_parallel_loops: " << _num_parallel_loops << std::endl;
        std::cout << "  dim_types: ";
        for (const auto& type : _dim_types) {
            std::cout << static_cast<int>(type) << " ";
        }
        std::cout << std::endl;
        std::cout << "  dim_sizes: ";
        for (const auto& size : _dim_sizes) {
            std::cout << size << " ";
        }
        std::cout << std::endl;
        std::cout << "  loop_order: ";
        for (const auto& size : _loop_order) {
            std::cout << size << " ";
        }
        std::cout << std::endl;
        for (size_t i = 0; i < _num_parallel_loops; i++) {
            size_parallel_loops *= _dim_sizes[_loop_order[i]];
        }

        std::cout << "size_parallel_loops: " << size_parallel_loops << std::endl;

        // #pragma omp parallel for
        for (int64_t it_all = 0; it_all < size_parallel_loops; it_all++) {
            int64_t it_remaining = it_all;
            std::cout << "init it_remaining: " << it_remaining << std::endl;

            const char* temp_ptr_in0 = static_cast<const char*>(ptr_in0);
            const char* temp_ptr_in1 = static_cast<const char*>(ptr_in1);
            char* temp_ptr_out = static_cast<char*>(ptr_out);

            for (int64_t id_loop = _num_parallel_loops - 1; id_loop >= 0; id_loop--) {
                // calculate loop index l_it for loop l_id_loop
                int64_t it = it_remaining % _dim_sizes[_loop_order[id_loop]];
                it_remaining = it_remaining / _dim_sizes[_loop_order[id_loop]];

                // update pointer with strides
                temp_ptr_in0 += it * _strides_in0[_loop_order[id_loop]] * 4;
                temp_ptr_in1 += it * _strides_in1[_loop_order[id_loop]] * 4;
                temp_ptr_out += it * _strides_out[_loop_order[id_loop]] * 4;
            }
            // call non parallel loops or kernel

            bool thread_first_access = first_access && (it_all == 0);
            bool thread_last_access = last_access && (it_all == size_parallel_loops - 1);

            execute_iter(_loop_order[_num_parallel_loops],
                         temp_ptr_in0,
                         temp_ptr_in1,
                         temp_ptr_out,
                         thread_first_access,
                         thread_last_access,
                         _num_parallel_loops);  // Added missing argument for loop_count
        }
    }

    void TensorOperation::print() {
        std::cout << "TensorOperation:" << std::endl;
        std::cout << "  dtype: " << static_cast<int>(_dtype) << std::endl;
        std::cout << "  prim_first_touch: " << static_cast<int>(_prim_first_touch) << std::endl;
        std::cout << "  prim_main: " << static_cast<int>(_prim_main) << std::endl;
        std::cout << "  prim_last_touch: " << static_cast<int>(_prim_last_touch) << std::endl;

        std::cout << "  dim_types: ";
        for (const auto& type : _dim_types) {
            std::cout << static_cast<int>(type) << " ";
        }
        std::cout << std::endl;

        std::cout << "  exec_types: ";
        for (const auto& type : _exec_types) {
            std::cout << static_cast<int>(type) << " ";
        }
        std::cout << std::endl;

        std::cout << "  dim_sizes: ";
        for (const auto& size : _dim_sizes) {
            std::cout << size << " ";
        }
        std::cout << std::endl;

        std::cout << "  strides_in0: ";
        for (const auto& stride : _strides_in0) {
            std::cout << stride << " ";
        }
        std::cout << std::endl;

        std::cout << "  strides_in1: ";
        for (const auto& stride : _strides_in1) {
            std::cout << stride << " ";
        }
        std::cout << std::endl;

        std::cout << "  strides_out: ";
        for (const auto& stride : _strides_out) {
            std::cout << stride << " ";
        }
        std::cout << std::endl;

        // Print loop sizes and order
        if (!_loop_sizes.empty()) {
            std::cout << "  loop_sizes: ";
            for (const auto& size : _loop_sizes) {
                std::cout << size << " ";
            }
            std::cout << "\n";

            if (!_loop_order.empty()) {
                std::cout << "  loop_order: ";
                for (const auto& order : _loop_order) {
                    std::cout << order << " ";
                }
                std::cout << "\n";
            }
        }
    }
}  // namespace einsum::backend
