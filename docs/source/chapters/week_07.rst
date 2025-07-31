
Tensor Operations
=================

Backend
-------

This week we move away from just-in-time generated code, and get closer to our tensor compiler.
The tensor backend serves as the next higher level of abstraction.
We use our code generator to support primitive operations and execute binary contractions quickly on the hardware. The main component that is added here are additional dimensions that we define as loops around the GEMM.
The goal is to use configurations like this, to let our backend generate an excution funktion that contracts the two input tensors to an output tensor with the execution types exactily as it specified here.

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



This table is a translation of the following einsum expression :code:`abdc, ebfd -> aefc`.
Since this step really only executes what is entered, we actually only need 4 functions. 
First the setup function takes all the entries made in the table above and saves them in a TensorOperation object.
Here you can see the main part of the setup function:

.. code-block:: C++
    :linenos:

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

The second function is already the compile function, in which the JIT kernel are generated and the compile time parameters are set.
The correct leading dimensions are also already determined from the strides array, even if they are passed to the kernel at runtime.

When setting the parameters for the kernels, we only have to make sure that the conditions are correct, i.e. that the left input and output have the same M dimension stride 1 and the right tensor has a K dimension stride 1.
The only special feature in our code is that if the post operation is a relu operation, we create a GEMM kernel as the last touch that also processes the ReLU fuset in the kernel.
Thus the kernel is called as the last touch and no last normal GEMM kernel is made previously.
Here you can understand this decision in the code:

.. code-block:: C++
    :linenos:

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

The tensor contraction is then processed with the help of the :code:`execute` function.
This is used as an entry point to then recursively call the :code:`execute_iter` function.
There, the previously determined loop dimensions are processed until you arrive at the primary function (GEMM).
The correct address within the tensor is calculated for each loop iteration and then the pointers can then be passed to the GEMM.
Here you can see the pointer calculation part of the :code:`execute_iter` function:

.. code-block:: C++
    :linenos:

    // update pointer with strides
    char* l_ptr_in0 = const_cast<char*>(ptr_in0);
    char* l_ptr_in1 = const_cast<char*>(ptr_in1);
    char* l_ptr_out = ptr_out;

    if (_loop_ids.size() > 0) {
        l_ptr_in0 += l_it * _strides_in0[_loop_ids[id_loop]] * 4;
        l_ptr_in1 += l_it * _strides_in1[_loop_ids[id_loop]] * 4;
        l_ptr_out += l_it * _strides_out[_loop_ids[id_loop]] * 4;
    }

This implementation now leads back to the example at the top that we executed with our TensorOperation backend.
Here is the result of the execution.
Since no parallel loops are specified in the example, these results are single threaded.

.. code-block:: text

    Running first example...
    Total error first example: 0
    Execution time for third example: 0.903002 seconds
    GFLOPS for third example: 118.676

The second setting we tested is the same as the first, but with 4 PRIM dimensions.
So we now use the BRGEMM as primitive operation.
There is only one line changed int the setup:

.. list-table:: 
   :widths: 40 60
   :header-rows: 1

   * - Variable
     - Value
   * - exec_types
     - (   Seq,  Seq,  Prim, Prim, Prim, Prim )

This is the performance we got for this example:

.. code-block:: text

      Running second example...
      Total error second example: 0
      Execution time for third example: 0.900494 seconds
      GFLOPS for third example: 119.006

As a final example, the last touch primitive is adapted so that an element-by-element ReLU is made after the contraction. 
These are our results:

.. code-block:: text

      Running third example with ReLU activation...
      Total error third example: 0
      Execution time for third example: 0.912997 seconds
      GFLOPS for third example: 117.377

To summarize, all three settings deliver good performance. T
The performance increase from the first to the second example is due to the fact that a larger K loop is now made in the primitive. 
In the third example, the performance drops only slightly, as there is additional overhead due to the ReLU.
Nevertheless, we are very satisfied that it is only 2 GFLOPS, we know that it was significantly more before the fused implementation.

To run our performance tests use the executable :code:`./build/bin/bench_ten_op_backend`, which can be run when you have built the project.



Unary
-----

We have decided to write a new class for the Unary tensor operations. This is very similar to the binary backend, except that it only gets one tensor as input.
There is also just one main primitive, no first or last touch primitive.
You can see the code in the file `TensorOperationUnary.cpp <https://github.com/stefan0re/machine_learning_compiler/blob/main/src/einsum/backend/TensorOperationUnary.cpp>`_.
For the reorder operation, we decided to execute scalar code because we did not have a fully functional transpose primitive at the time of development.