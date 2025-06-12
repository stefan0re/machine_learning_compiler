
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
        std::vector<uint32_t> notation;
        node_t node_type;
        TreeNode* parent;
        TreeNode* left_child;
        TreeNode* right_child;
    };

The einsum notation for the output tensor of each node is stored as a vector of integer in each respective node.
The tree is created by iterating over each character of the input string and storing the context in the before-mentioned stack. Based on this information we decide, where a new node is supposed to be added.

By utilizing a recursive function to print the tree we can see, that a correct tree was build:

.. code-block:: text

  └─ 0,2
     ├─ 0,1
     └─ 1,2

This also works for larger trees:

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

Coming to the lowering function, we first wanted to implement a working function for with only contraction nodes as our unary operations are not working correctly.
We solve this again by using a structure, that hold both the operation objects and the corresponding tensor information:

.. code-block:: C++
    :linenos:

    struct OpSteps {
        struct OpStep {
            uint32_t in_ten_left;
            std::vector<uint32_t> in_ten_left_notation;
            uint32_t in_ten_right;
            std::vector<uint32_t> in_ten_right_notation;
            uint32_t out_ten;
            std::vector<uint32_t> out_ten_notation;
        };
        std::vector<einsum::backend::TensorOperation> step_list;
        std::vector<OpStep> tensor_order;
    };

We solve this task by using a recursive function, that returns a :code:`OpStep` object. If we find our selfs in a contraction node, which is marked by the type

.. code-block:: C++
    :linenos:

    enum class node_t : uint32_t {
        leaf = 0,
        permutation = 1,
        contraction = 2,
    };

we first get the information of each child (and maybe even create another operation by doing so). Then create a :code:`TensorOperation` object with this collected information.
Finally we add this object and a :code:`OpStep` object to the respective lists in :code:`OpSteps`.

Our code fail when adding the :code:`TensorOperation` object to the vector. As our implementation has no copy constructor. This error is going to be fixed in near future.
As we do not have a fully working version, we have no benchmarks to report.

We all worked on the tasks in equal parts.
Our code can be viewed on `Github <https://github.com/stefan0re/machine_learning_compiler>`_ under version week9.
