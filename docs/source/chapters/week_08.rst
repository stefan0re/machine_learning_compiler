
Tensor Operations
=================

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



Our code can be viewed on `Github <https://github.com/stefan0re/machine_learning_compiler>`_ under version week8.
