
Optimization Passes
===================

Up front, the optimizations are implemented in the order in which they are explained here.
First, fusion to have more options for splitting.
Then identifying the primitives and thereafter the reordering of the loop dimensions.
We will discuss parallelization separately.

Dimension fusion
----------------

With dimension fusing, we try to merge unsuitable dimensions that are too small with adjacent dimensions of the same type. To eliminate the small dimension size.
For all three dimension types M, N and K we have set a threshold of 64, if a dimension is smaller than this limit an attempt is made to fuse the dimension.
Here is an example of how it is implemented for the M dimension:

.. code-block:: C++

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

Dimension splitting
-------------------

Dimension splitting is exactly the opposite of fusing as you try to reduce dimensions that are too large.
To split a large dimension, we break it down into its prime factors and then find a size of approximately 64 for one dimension.
The limit at which we try to split a dimension is 128.
For example, if we have a dimension of size 1600, we split it into 25 and 64.

Hier ist noch einmal die Umsetzung für die Dimension M:

.. code-block:: C++

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


Primitive identification
------------------------

This optimization step takes over the task of the user, who should actually specify with his configuration object what the primitive dimensions and how the parallelization should run.
But if none of this is defined, this identification function finds the correct primitives and defines both a simple loop sequence and where to parallelize.
Thus, for a primitive M dimension, the dimension is searched which has stride 1 in both the left and the output tensor.
For the K dimension, the stride 1 dimension in the right output tensor is searched and for the BR_K dimension the following K dimension.
For the N dimension, we simply search for the N dimension in the right tensor with the lowest stride.

Dimension reordering
--------------------

To reorder the loop dimensions we use the simple heuristic from the lecture, hence we alternate the dimensions types.
To do this, we rewrite the loop_order array and put them in a good order.
This code example from our reorder function should make it clear how it is achieved.

.. code-block:: C++

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

Shared memory parallelization
-----------------------------

We now have loops around our GEMM, so it makes sense to parallelize them with OpenMP to take advantage of the fact that we don't just have one core on the CPU.
For simplicity, we have implemented it that if the outermost loop is an M or N dimension, it is parallelized with OpenMP.
That's why our :code:`execute_iter_parallel` function actually looks exactly like the :code:`execute_iter` function only with an OpenMP pragma above the loop.
In our experiments we couldn't see a big problem with this implementation because our CPU just have 4 core, but we are aware that in some cases our code isn't able to parallize properly.

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

Result:

.. code-block:: text

  Running first example with optimizations...
    Total error first example: 0
    Execution first for third example: 1.19062 seconds
    GFLOPS for first example: 343.915

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


Result:

.. code-block:: text

  Running second example with optimizations...
    Total error second example: 0
    Execution time for second example: 1.24799 seconds
    GFLOPS for second example: 328.106

From our final optimized config we could read out that in the first case the primitive M dimension has the size 64 and in the second case it has only 25.
Since our optimization considers the sizes to be neither too large nor too small, everything remains the same.
This is the reason for the difference in performance between the two examples.

Finally, we tested our own example, which is a little smaller:

 .. list-table:: Tensor contraction example.
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Value
   * - dim_types
     - (   M,    M,     N,    N,     K )
   * - exec_types
     - ( Seq,  Seq,   Seq,  Seq,   Seq )
   * - dim_sizes
     - (   6,    8,    10,   12,    16 )
   * - strides_in0
     - (   1,    6,     0,    0,    48 )
   * - strides_in1
     - (   0,    0,    16,  160,     1 )
   * - strides_out 
     - (   1,    6,    48,  480,     0 )

Result:

.. code-block:: text
  
  Running own example with optimizations...
    Total error own example: 0
    Execution time for own example: 3.44223 seconds
    GFLOPS for own example: 51.8734

In general, this contraction has a very low arithmetic intensity.
The results of this example are significantly worse than the previous ones, because on the one hand the outer loops are small and the parallelization therefore brings more overhead than benefit.
And on the other hand, the kernel sizes are small even if they have already been fused.

The executable for these benchmarks is again in the build directory and can be run with the command :code:`./build/bin/bench_ten_op_optimized`.