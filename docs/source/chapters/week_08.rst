
Optimization Passes
===================

Dimension Splitting
-------------------

To fuse dimensions we set at threshold of 64. Each dimension no matter if it is M, N or K is split if its size is larger than 64.
Therefore, we calculate the primfactorization of the dimension size and then multply the factors until the biggest size smaller than 64 is reached.

In this code snippet, we implement the dimension splitting for the M dimension:

.. code-block:: C++
    :linenos:

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

As you can see, at the point where the new sizes are set, we also set the strides for the input and output tensors. The new size is multiplied with the stride of the original dimension to ensure that the virtual memory layout remains correct.

Dimension fusion
----------------

To fuse dimensions we set a threshold of 16. Each dimension that is smaller then 16 is looking for an appropriate dimension to fuse with.
Therefore, we look for the next surrounding dimension that is of the same type.
If there is one we fuse the two dimensions by multiplying their sizes and setting the stride of the new dimension to the sum of the two strides.

Here you can see our code implementation for the M dimension fusion:

.. code-block:: C++
    :linenos:

        std::vector<int64_t> m_loop_ids;
        for (size_t i = 0; i < _dim_types.size(); i++) {
            if (_dim_types[i] == dim_t::m) {
                m_loop_ids.push_back(i);
            }
        }

        for (size_t i = 0; i < m_loop_ids.size(); i++) {
            if (_dim_sizes[m_loop_ids[i]] < 16) {

                if (i + 1 < m_loop_ids.size() && _dim_types[m_loop_ids[i + 1]] == dim_t::m) {
                    // fuse the two M dimensions
                    _dim_sizes_storage[m_loop_ids[i]] *= _dim_sizes[m_loop_ids[i + 1]];

                    // remove the next M dimension
                    _dim_types_storage.erase(_dim_types_storage.begin() + m_loop_ids[i + 1]);
                    _dim_sizes_storage.erase(_dim_sizes_storage.begin() + m_loop_ids[i + 1]);

                    _strides_in0_storage[m_loop_ids[i]] *= _dim_sizes[m_loop_ids[i + 1]];
                    _strides_in0_storage.erase(_strides_in0_storage.begin() + m_loop_ids[i + 1]);

                    _strides_out_storage[m_loop_ids[i]] *= _dim_sizes[m_loop_ids[i + 1]];
                    _strides_out_storage.erase(_strides_out_storage.begin() + m_loop_ids[i + 1]);
                }
            }
        }
  
As a disclaimer this function isn't really well test in practice therefore we started investigating into bigger contractions with more dimension. 
Also important to say, this function is called before the split optimization, therefore we have more options to split into nice dimensions.

Primitive identification
------------------------

Before the correct loop sequence can be defined, the execution types are now defined. To find the M dimension for the BRGEMM, we simply search for the stride an M dimension in the left input tensor and in the output tensor.
For the K dimension we do the same only this time in the right and in the output tensor.
For the N dimension we select the one with the minimal stride, so we can be sure that the GEMM dimensions are not too far apart in memory.
As BR dimension we then look for a second K dimension and again take the one with the smallest stride.
If we do not find a stride 1 dimension for M and K an error is thrown.
The implementation of this looks pretty boring, which is why we won't show it here. :D

Dimension reordering
--------------------

For the dimensions reordering we have decided to limit ourselves to the M and N dimensions only.
Here we have applied the simple heuristic that as soon as two of the same dimension type are next to each other we try to swap one dimension with another dimension type.
We hope that this will improve the cache efficiency.
In our dimensions, we only go through those that were previously marked as loop dimensions.

Here you can see our simple code implementation:

.. code-block:: C++
    :linenos:

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

This optimization step also sets the loop sequence for later execution.


Shared Memory Parallelization
-----------------------------

The code for our Shared Memory Parallelization does work, for us to find the bug, we will need further investigation. At the moment the code does just nothing to the output tensor.

.. code-block:: C++
    :linenos:

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




Test Setups
-----------

The first setup that we tryied was a simple matrix multiplication with the following dimensions and strides:

.. list-table:: Matrix multiplication example.
   :widths: 40 60
   :header-rows: 1

   * - Variable
     - Value
   * - dim_types
     - (    M,    N,    K )
   * - exec_types
     - (  Seq,  Seq,  Seq )
   * - dim_sizes
     - ( 1600, 1600, 1600 )
   * - strides_in0
     - (    1,    0, 1600 )
   * - strides_in1
     - (    0, 1600,    1 )
   * - strides_out
     - (    1, 1600,    0 )

Here optimization split all dimensions, because they are way to big for our kernel generator.
Then we identified the Primitive dimensions and the loop dimensions.
And finally we reordered the M and N dimension.
Here you can see the resulting setup that was printed by our code:

.. code-block:: text
    :linenos:

    Testing Setting 1
    ***********************
    TensorOperation setup:
    dtype: 0
    prim_first_touch: 99
    prim_main: 3
    prim_last_touch: 99
    id_first_primitive_loop: 0
    id_prim_m: 1
    id_prim_n: 2
    id_prim_k: 4
    id_prim_br: 5
    strides_in0: 64 1 0 0 102400 1600 
    strides_in1: 0 0 1600 102400 1 64 
    strides_out: 64 1 102400 1600 0 0 
    dim_types: 1 1 2 2 3 3 
    dim_sizes: 25 64 64 25 64 25 
    exec_types: 0 1 1 0 1 1 
    loop_sizes: 25 1 1 25 1 1 
    loop_order: 3 0 
    lda: 1600
    ldb: 1600
    ldc: 1600
    in0_br_stride: 1600
    in1_br_stride: 64
    Setting 1 completed.
    ************************
        

The second setup that we tried was a tensor contraction with the following dimensions and strides:

.. list-table:: Tensor contraction example.
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Value
   * - dim_types
     - (   M,    M,     N,    N,     K,    K )
   * - exec_types
     - ( Seq,  Seq,   Seq,  Seq,   Seq,  Seq )
   * - dim_sizes
     - (  64,   25,    64,   25,    64,   25 )
   * - strides_in0
     - (  25,    1,     0,    0, 40000, 1600 )
   * - strides_in1
     - (   0,    0, 40000, 1600,    25,    1 )
   * - strides_out
     - (  25,    1, 40000, 1600,     0,    0 )

Here our optimization coulnt do much, because the dimensions are already small enough.
And the M and N dimensions are already next to each other.

The resulting setup that was printed by our code:

.. code-block:: text
    :linenos:

    Testing Setting 2
    ***********************
    TensorOperation setup:
    dtype: 0
    prim_first_touch: 99
    prim_main: 3
    prim_last_touch: 99
    id_prim_m: 1
    id_prim_n: 3
    id_prim_k: 5
    id_prim_br: 4
    strides_in0: 25 1 0 0 40000 1600 
    strides_in1: 0 0 40000 1600 25 1 
    strides_out: 25 1 40000 1600 0 0 
    dim_types: 1 1 2 2 3 3 
    dim_sizes: 64 25 64 25 64 25 
    exec_types: 0 1 0 1 1 1 
    loop_sizes: 64 1 64 1 1 1 
    loop_order: 2 0 
    lda: 1600
    ldb: 1600
    ldc: 1600
    in0_br_stride: 40000
    in1_br_stride: 25
    Setting 2 completed.
    ************************

And this is the third setup that we tried, which is a big tensor contraction with the following dimensions and strides:


.. list-table:: Big Tensor contraction example.
  :widths: 30 70
  :header-rows: 1

  * - Variable
    - Value
  * - dim_types
    - (   M,    M,    M,     N,    N,    N,     K,     K,     K )
  * - exec_types
    - ( Seq,  Seq,  Seq,   Seq,  Seq,  Seq,   Seq,   Seq,   Seq )
  * - dim_sizes
    - (   2,    4,   48,     3,    7,   64,    16,    16,    96 )
  * - strides_in0
    - ( 192,   48,    1,     0,    0,    0,   384,  6144, 98304 )
  * - strides_in1
    - (   0,    0,    0, 2064384, 688128, 98304, 1536,   96,     1 )
  * - strides_out
    - ( 192,   48,    1, 258048, 86016, 12288,     0,     0,     0 )
  
Unfortunately, this large tensor contraction has shown us the limits of our implementation.
This is because an unauthorized memory access has occurred.

Benchmarks
----------

.. make a small table for setup 1 and two to show GFLOPS and measured time

.. list-table:: Benchmarks for setup 1 and 2.
   :widths: 20 20 20
   :header-rows: 1

   * - Setup
     - GFLOPS
     - Time (ms)
   * - Setup 1
     - 57.55
     - 14.23
   * - Setup 2
     - 40.4211
     - 20.2666



Our code can be viewed on `Github <https://github.com/stefan0re/machine_learning_compiler>`_ under version week8.
