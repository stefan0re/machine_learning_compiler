
Einsum Trees
============

This week we set the basics for working with einsum trees. We implemented both a parsing algorithm and a lowering algorithm.

Lowering
--------

Before lowering a tree to the backend, we have to create a tree from a string. For example:

.. code-block:: C++

    "[0,1],[1,2]->[0,2]"

With the help of a stack we translate this into a tree structure with nodes:

.. code-block:: C++
    :linenos:

    struct TreeNode {
        int32_t id;
        node_t node_type;
        TreeNode* parent;
        TreeNode* left_child;
        TreeNode* right_child;

        std::vector<uint32_t> notation;
        Tensor* left_tensor;
        Tensor* right_tensor;
        Tensor* out_tensor;

        TensorOperation::prim_t first_touch;
        TensorOperation::prim_t operation_primitive;
        TensorOperation::prim_t last_touch;

        TensorOperation op;
    };

The einsum notation for the output tensor of each node is stored as a vector of integer in each respective node.
The tree is created by iterating over each character of the input string and storing the context in the before-mentioned stack. Based on this information we decide, where a new node is supposed to be added.

Each node has a type, which defines if the node is a leaf, a permutation or a contraction node. The stride information is calculated in the separate Tensor class, which is not shown here.
By utilizing a recursive function to print the tree we can see, that a correct tree was build:

.. code-block:: text

  └─ 0,2
     ├─ 0,1
     └─ 1,2

This also works for larger trees:

Example 1
^^^^^^^^^

.. code-block:: C++

    "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]"

.. code-block:: text

  └─ 0,1,2,3,4
     ├─ 7,3,4
     │  ├─ 8,4
     │  └─ 7,3,8
     └─ 0,1,2,7
        ├─ 1,2,5,7
        │  ├─ 2,6,7
        │  └─ 1,5,6
        └─ 0,5

And with permutation nodes:

Example 2
^^^^^^^^^	

.. code-block:: C++

    "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]"

.. code-block:: text

    └─ 0,1,2,3
       ├─ 0,4,7,8,2,3
       │  ├─ 7,8,5,6,2,3
       │  │  ├─ 8,6,9,3
       │  │  │  └─ 3,6,8,9
       │  │  └─ 7,5,2,9
       │  │     └─ 2,5,7,9
       │  └─ 0,4,5,6
       └─ 1,4,7,8

The lowering function is implemented in a recursive manner. It moves downwards through the tree and creates the :code:`TensorOperation` and :code:`TensorOperationUnary` objects for each node that are later used to execute the tree.
The information to create the operation objects is taken from the size vector and each tensor class of the node.

Benchmarks
^^^^^^^^^^

.. list-table:: Benchmarks for compiling and executing einsum trees
   :widths: 40 30 30
   :header-rows: 1

   * - Variable
     - Compile Time (ms)
     - GFLOPS
   * - Example 1
     - 0.21
     - 6.00
   * - Example 2
     - -
     - -

We all worked on the tasks in equal parts.
Our code can be viewed on `Github <https://github.com/stefan0re/machine_learning_compiler>`_ under version week9.
