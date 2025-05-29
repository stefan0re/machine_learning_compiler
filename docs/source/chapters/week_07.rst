
Tensor Operations
=================

Backend
-------

This week we move away from just-in-time generated code, and get closer to our tensor compiler.
The goal is to use configurations like this: 

.. list-table:: 
   :widths: 40 60
   :header-rows: 1

   * - Variable
     - Value
   * - dtype
     - FP32
   * - prim_first_touch
     - None
   * - prim_main
     - GEMM
   * - prim_last_touch
     - None
   * - dim_types
     - (     M,    N,    K,    M,    N,    K )
   * - exec_types
     - (   Seq,  Seq,  Seq, Prim, Prim, Prim )
   * - dim_sizes
     - (    32,   32,    8,   32,   32,   32 )
   * - strides_in0
     - (  8192,    0, 1024,    1,    0,   32 )
   * - strides_in1
     - (     0, 8192, 1024,    0,   32,    1 )
   * - strides_out
     - ( 32768, 1024,    0,    1,   32,    0 )

To generate compute the binary tensor contraction which can also be written in einsum as: 

:code:`abdc, ebfd -> aefc`

To pass this configuration to the binary contraction generator we use this setup method:


.. code-block:: C++
    :linenos:

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

        // write loop dims to _loop_sizes_storage
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_exec_types[i] != exec_t::prim) {
                _loop_sizes_storage.push_back(_dim_sizes[i]);
            } else {
                _id_first_primitive_loop = i;
                break;
            }
        }
        _loop_sizes = _loop_sizes_storage;

        // set prim ids
        for (size_t i = 0; i < _dim_sizes.size(); i++) {
            if (_dim_types[i] == dim_t::m && _exec_types[i] == exec_t::prim) {
                _id_prim_m = i;
            } else if (_dim_types[i] == dim_t::n && _exec_types[i] == exec_t::prim) {
                _id_prim_n = i;
            } else if (_dim_types[i] == dim_t::k && _exec_types[i] == exec_t::prim) {
                if (_id_prim_k == 0) {
                    _id_prim_k = i;
                } else {
                    _id_prim_br_size = _id_prim_k;
                    _id_prim_k = i;
                }
            }
        }

        // create brgemm_kernel
        _brgemm.generate(_dim_sizes[_id_prim_m],
                        _dim_sizes[_id_prim_n],
                        _dim_sizes[_id_prim_k],
                        (_id_prim_br_size > -1) ? _dim_sizes[_id_prim_br_size] : 1,
                        0,
                        0,
                        0,
                        static_cast<mini_jit::generator::Brgemm::dtype_t>(_dtype));
        _brgemm_kernel = _brgemm.get_kernel();

        // set lda, ldb, ldc, in0_br_stride, in1_br_stride
        // TODO: currently assumes primitve types are always the last 3 dimensions
        _lda = _strides_in0[_strides_in0.size() - 1];
        _ldb = _strides_in1[_strides_in1.size() - 2];
        _ldc = _strides_out[_strides_out.size() - 2];

        _in0_br_stride = _strides_in0[_strides_in0.size() - 4];
        _in1_br_stride = _strides_in1[_strides_in1.size() - 4];

        return error_t::success;
    }


In the upper part, the object variables are set first, then from line 35 the loop dimension are detected.
The IDs of the primitive dimensions are then identified.
At the end, the JIT kernel (BRGEMM) is generated. In future, the unary kernel for first and last touch should be generated here.


To create the loops around the BRGEMM kernel, we have the function :code:`execute_iter` which calls itself recursively until a primitive dimension is reached. 
Here the recursion breaks off and our BR_GEMM kernel is called.
In order to pass the correct addresses here, the correct stride for the respective loop is calculated beforehand for the respective tensor. 

.. code-block:: C++
    :linenos:
    
    void TensorOperation::execute_iter(int64_t id_loop,
                                       char const* ptr_in0,
                                       char const* ptr_in1,
                                       char* ptr_out,
                                       bool first_access,
                                       bool last_access) {
        int64_t l_size = _loop_sizes[id_loop];

        for (int64_t l_it = 0; l_it < l_size; l_it++) {
            char* l_ptr_in0 = const_cast<char*>(ptr_in0) + l_it * _strides_in0[id_loop] * 4;
            char* l_ptr_in1 = const_cast<char*>(ptr_in1) + l_it * _strides_in1[id_loop] * 4;
            char* l_ptr_out = ptr_out + l_it * _strides_out[id_loop] * 4;

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
                    _unary_first_touch_kernel(l_ptr_in0, l_ptr_out, _ldc, _ldc);
                }
                _brgemm_kernel(l_ptr_in0, l_ptr_in1, l_ptr_out,
                               _lda,
                               _ldb,
                               _ldc,
                               _in0_br_stride,
                               _in1_br_stride);

                // handle last touch
                if (last_access && _prim_last_touch != prim_t::none) {
                    _unary_last_touch_kernel(l_ptr_out, l_ptr_out, _ldc, _ldc);
                }
            }
        }
    }

Currently the first and last access are not working correctly, but in the future they should be used to call the unary kernels for first and last touch.
The rest of our code calculates correctly, but the performance is unfortunately only below 10 GFLOPS for all three implementations.
This indicates that we still have to work on our BRGEMM generator, because it only gives a peak performance of 60 GFLOPS for super sized matrices. 
We have also tried other settings and found that the primitive dimensions are the decisive ones, if we get large dimension sizes here, our einsum implementation also performs better.

Our code can be viewed on `Github <https://github.com/stefan0re/machine_learning_compiler>`_ under version week8.
