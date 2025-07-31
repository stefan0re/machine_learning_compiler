Einsum Tree Optimization
========================

This week we focused on optimizing einsum trees, specifically by swapping and inserting permutations to improve the efficiency of tensor contractions. 
We did not implement reorder, as this already exists in the backend. We implemented a function that that first looks for a possible swap and afterwards checks if permutations are needed.

Swapping Nodes
--------------

First we need a function that scores the current child configuration and the possible swapped configuration. This is achieved how far a dimension of interest is away from its desired position. The further away a dimension is from its desired position, the higher the score.

.. code-block:: C++

    double EinsumTree::getScore(TreeNode* node, TensorOperation::dim_t dim_type, bool swap) {
        TensorOperation::dim_t left_dim_type, right_dim_type;
        if (swap) {
            left_dim_type = TensorOperation::dim_t::k;
            right_dim_type = dim_type;
        } else {
            left_dim_type = dim_type;
            right_dim_type = TensorOperation::dim_t::k;
        }

        double score = 0;
        double left_score = 0.f;
        double right_score = 0.f;
        double out_score = 0.f;
        double num_left = 0.f;
        double num_right = 0.f;
        double num_out = 0.f;

        for (size_t i = 0; i < node->left_child->notation.size(); i++) {
            if (node->left_tensor->id[i].dim_t == static_cast<int>(left_dim_type)) {
                left_score += static_cast<double>(node->left_child->notation.size() - (i + 1));
                num_left += 1;
            }
        }
        score += left_score / num_left;
        for (size_t i = 0; i < node->right_child->notation.size(); i++) {
            if (node->right_tensor->id[i].dim_t == static_cast<int>(right_dim_type)) {
                right_score += static_cast<double>(node->right_child->notation.size() - (i + 1));
                num_right += 1;
            }
        }
        score += right_score / num_right;
        for (size_t i = 0; i < node->notation.size(); i++) {
            if (node->out_tensor->id[i].dim_t == static_cast<int>(dim_type)) {
                out_score += static_cast<double>(node->notation.size() - (i + 1));
                num_out += 1;
            }
        }
        score += out_score / num_out;
        return score;
    }

Next we implement the swap function, which swaps the left and right child of a contraction node and also swaps the dimension types of the tensors involved. Furthermore, we adjust the dimension types of the output tensor accordingly.

.. code-block:: C++

    void EinsumTree::swap(TreeNode* parent) {
        if (parent == nullptr) {
            std::cerr << "Parent node is null, cannot swap." << std::endl;
            return;
        }

        if (parent->node_type != node_t::contraction) {
            std::cerr << "Swap can only be performed on contraction nodes." << std::endl;
            return;
        }

        // Swap left and right children
        TreeNode* temp = parent->left_child;
        Tensor* temp_tensor = parent->left_tensor;
        parent->left_child = parent->right_child;
        parent->left_tensor = parent->right_tensor;
        parent->right_child = temp;
        parent->right_tensor = temp_tensor;

        // Swap m and n dimension types in the tensors
        for (size_t i = 0; i < parent->left_tensor->id.size(); i++) {
            if (parent->left_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::m)) {
                parent->left_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::n);
            } else if (parent->left_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::n)) {
                parent->left_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::m);
            }
        }

        for (size_t i = 0; i < parent->right_tensor->id.size(); i++) {
            if (parent->right_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::m)) {
                parent->right_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::n);
            } else if (parent->right_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::n)) {
                parent->right_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::m);
            }
        }

        for (size_t i = 0; i < parent->out_tensor->id.size(); i++) {
            if (parent->out_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::m)) {
                parent->out_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::n);
            } else if (parent->out_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::n)) {
                parent->out_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::m);
            }
        }
    }

Lastly, in our main optimization function, we can now check if a swap is beneficial by comparing the scores of the current configuration and the swapped configuration. If the swapped configuration has a lower score, we perform the swap.

.. code-block:: C++

    double current_score = getScore(node, TensorOperation::dim_t::m, false);
    double swap_score = getScore(node, TensorOperation::dim_t::n, true);
    if (swap_score < current_score) {

        swap(node);
    }

Insert Permutations
-------------------
In order to optimize the einsum tree further, we need to insert permutation nodes where necessary. First, we define a function to insert a permutation node into the tree.

.. code-block:: C++

    void EinsumTree::insertPermutation(TreeNode* parent, TreeNode* new_child, bool is_left) {
        if (parent == nullptr || new_child == nullptr) {
            std::cerr << "Parent or new child node is null, cannot insert." << std::endl;
            return;
        }

        if (new_child->node_type != node_t::permutation) {
            std::cerr << "New child must be a permutation node." << std::endl;
            return;
        }
        TreeNode* temp = nullptr;
        if (is_left) {
            new_child->id = this->size;
            this->size++;

            temp = parent->left_child;
            parent->left_child = new_child;
            parent->left_tensor = new_child->out_tensor;
            new_child->parent = parent;
            if (temp != nullptr) {
                new_child->left_child = temp;
                new_child->left_tensor = temp->left_tensor;
                temp->parent = new_child;
            }
        } else {
            if (parent->right_child != nullptr) {
                std::cerr << "Right child already exists, cannot insert new permutation node." << std::endl;
                return;
            }
            new_child->id = this->size;
            this->size++;

            temp = parent->right_child;
            parent->right_child = new_child;
            parent->right_tensor = new_child->out_tensor;
            new_child->parent = parent;
            if (temp != nullptr) {
                new_child->left_child = temp;
                new_child->left_tensor = temp->left_tensor;
                temp->parent = new_child;
            }
        }
    }

Next, we continue to implement the optimization function. Each contraction node is checked for the right order of `m` and `k` dimensions in the left tensor and `n` and `k` dimensions in the right tensor. If necessary, permutation nodes are inserted to ensure that the dimensions are in the correct order for efficient contraction.

.. code-block:: C++

    // Insert left permutation node if necessary
    bool add_left_permutation = false;
    std::vector<uint32_t> m_dim = {};
    std::vector<uint32_t> k_dim = {};
    for (size_t i = 0; i < node->left_tensor->id.size(); i++) {
        if (node->left_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::m)) {
            m_dim.push_back(node->left_child->notation[i]);
        } else if (node->left_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::k)) {
            k_dim.push_back(node->left_child->notation[i]);
        }
        for (size_t j = node->left_tensor->id.size() - 1; j > i; j--) {
            if (node->left_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::m) && node->left_tensor->id[j].dim_t == static_cast<int>(TensorOperation::dim_t::k)) {
                add_left_permutation = true;
                break;
            }
        }
    }

    size_t k_dim_size = k_dim.size();
    k_dim.insert(k_dim.end(), m_dim.begin(), m_dim.end());
    std::vector<uint32_t> new_notation = k_dim;
    if (add_left_permutation) {
        std::vector<uint32_t> out_dims;
        for (uint32_t dim_id : new_notation) {
            out_dims.push_back(this->id_dims[dim_id]);
        }
        Tensor* out_tensor = new Tensor(out_dims);

        for (size_t i = 0; i < new_notation.size(); i++) {
            out_tensor->id[i].exec_t = static_cast<int>(TensorOperation::exec_t::seq);
            if (i < k_dim_size) {
                out_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::k);
            } else {
                out_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::m);
            }
        }

        TreeNode* new_left_child = new TreeNode{
            0,                                // id
            EinsumTree::node_t::permutation,  // node_type
            node,                             // parent
            nullptr,                          // left_child
            nullptr,                          // right_child
            new_notation,                     // notation
            nullptr,                          // left_tensor
            nullptr,                          // right_tensor
            out_tensor,                       // out_tensor
            TensorOperation(),                // op
        };
        insertPermutation(node, new_left_child, true);
    }

    // insert right permutation node if necessary
    bool add_right_permutation = false;
    std::vector<uint32_t> n_dim = {};
    k_dim = {};

    for (size_t i = 0; i < node->right_tensor->id.size(); i++) {
        if (node->right_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::n)) {
            n_dim.push_back(node->right_child->notation[i]);
        } else if (node->right_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::k)) {
            k_dim.push_back(node->right_child->notation[i]);
        }
        for (size_t j = node->right_tensor->id.size() - 1; j > i; j--) {
            if (node->right_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::k) && node->right_tensor->id[j].dim_t == static_cast<int>(TensorOperation::dim_t::n)) {
                add_right_permutation = true;
                break;
            }
        }
    }

    size_t n_dim_size = n_dim.size();
    n_dim.insert(n_dim.end(), k_dim.begin(), k_dim.end());
    new_notation = n_dim;
    if (add_right_permutation) {
        std::vector<uint32_t> out_dims;
        for (uint32_t dim_id : new_notation) {
            out_dims.push_back(this->id_dims[dim_id]);
        }

        Tensor* out_tensor = new Tensor(out_dims);

        for (size_t i = 0; i < new_notation.size(); i++) {
            out_tensor->id[i].exec_t = static_cast<int>(TensorOperation::exec_t::seq);
            if (i < n_dim_size) {
                out_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::n);
            } else {
                out_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::k);
            }
        }

        TreeNode* new_right_child = new TreeNode{
            0,                                // id
            EinsumTree::node_t::permutation,  // node_type
            node,                             // parent
            nullptr,                          // left_child
            nullptr,                          // right_child
            new_notation,                     // notation
            nullptr,                          // left_tensor
            nullptr,                          // right_tensor
            out_tensor,                       // out_tensor
            TensorOperation(),                // op
        };
        insertPermutation(node, new_right_child, false);
    }

    // Recursively optimize left and right children
    optimizeNode(node->left_child);
    optimizeNode(node->right_child);

The function recursively continues to optimize the left and right children of the current node.

Results
-------

We tested our implementation with the given einsum strings.

Tree 1
^^^^^^
String ``[[7,3,8],[8,4]->[7,3,4]],[[0,5],[[5,1,6],[6,2,7]->[5,1,2,7]]->[0,1,2,7]]->[0,1,2,3,4]``
Tree structure:

.. code-block:: text

  └─ 0,1,2,3,4
     ├─ 7,3,4
     |  ├─ 7,3,8
     |  └─ 8,4
     └─ 0,1,2,7
        ├─ 0,5
        └─ 5,1,2,7
            ├─ 5,1,6
            └─ 6,2,7

Optimized:

.. code-block:: text

  └─ 0,1,2,3,4
     ├─ 7,3,4
     │  ├─ 8,4
     │  └─ 7,3,8
     └─ 0,1,2,7
        ├─ 5,1,2,7
        │  ├─ 6,2,7
        │  └─ 5,1,6
        └─ 0,5

Tree 2
^^^^^^
String ``[1,4,7,8],[[0,4,5,6],[[2,5,7,9],[3,6,8,9]->[2,5,7,3,6,8]]->[0,4,2,7,3,8]]->[0,1,2,3]``
Tree structure:

.. code-block:: text

  └─ 0,1,2,3
     ├─ 1,4,7,8
     └─ 0,4,2,7,3,8
        ├─ 0,4,5,6
        └─ 2,5,7,3,6,8
           ├─ 2,5,7,9
           └─ 3,6,8,9

Optimized:

.. code-block:: text

  └─ 0,1,2,3
     ├─ 4,7,8,0,2,3
     │  └─ 0,4,2,7,3,8
     │     ├─ 5,6,2,7,3,8
     │     │  └─ 2,5,7,3,6,8
     │     │     ├─ 9,3,6,8
     │     │     │  └─ 3,6,8,9
     │     │     └─ 2,5,7,9
     │     └─ 0,4,5,6
     └─ 1,4,7,8

Tree 3
^^^^^^
String ``[[0,1],[1,2]->[0,2]],[[3,4],[4,5]->[3,5]]->[0,1,2,3,4]``

Tree structure:

.. code-block:: text

  └─ 5,6,7,8,9
     ├─ 2,7,8,4
     │  ├─ 2,7,3
     │  └─ 3,8,4
     └─ 4,9,5,6,2
        ├─ 4,9,0
        └─ 0,5,6,2
           ├─ 0,5,1
           └─ 1,6,2

Optimized:

.. code-block:: text

  └─ 5,6,7,8,9
     ├─ 2,4,7,8
     │  └─ 2,7,8,4
     │     ├─ 3,8,4
     │     └─ 2,7,3
     └─ 4,9,5,6,2
        ├─ 0,5,6,2
        │  ├─ 1,6,2
        │  └─ 0,5,1
        └─ 4,9,0

Benchmarks
^^^^^^^^^^

.. list-table:: Benchmarks for compiling and executing einsum trees
   :widths: 40 30 30
   :header-rows: 1

   * - Variable
     - Compile Time (ms)
     - GFLOPS
   * - Example 1
     - 0.34
     - 3.01
   * - Example 2
     - -
     - -
   * - Example 3
     - -
     - -
   
We all worked on the tasks in equal parts.
Our code can be viewed on `Github <https://github.com/stefan0re/machine_learning_compiler>`_ under version week10.
