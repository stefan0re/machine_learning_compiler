#include "./einsum_trees.h"

#include <cstring>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "../../tensor/tensor.h"

using namespace einsum::trees;
using namespace einsum::backend;

EinsumTree::EinsumTree(std::string str_repr, std::vector<uint32_t> id_dims, bool use_bias) {
    // set class attribures
    this->id_dims = id_dims;
    this->use_bias = use_bias;  // flag for using biases

    /*
        Stack shows the state of parsing.
        'l': a left child was added last iteration
        'r': next child to be added has to be right
        'd': a right child was added last iteration
        'u': go one step up (to parent)
        'w': write into current node
    */

    // initialize not empty stack
    std::vector<char> stack = {'l'};

    // intitialize the root node
    this->root = new TreeNode{
        static_cast<int32_t>(this->size),  // id
        EinsumTree::node_t::leaf,          // node_type
        nullptr,                           // parent
        nullptr,                           // left_child
        nullptr,                           // right_child
        {},                                // notation
        nullptr,                           // left_tensor
        nullptr,                           // right_tensor
        nullptr,                           // out_child
        TensorOperation::prim_t::none,     // first touch
        TensorOperation::prim_t::none,     // operation primitive
        TensorOperation::prim_t::none,     // last touch
        TensorOperation(),                 // op
    };
    this->leaf_ids.push_back(static_cast<int32_t>(this->size));
    this->size++;

    TreeNode* current = this->root;

    // loop over all characters in string representation
    for (char character : str_repr) {
        if (character == '[') {
            if (stack.back() == 'l' || stack.back() == 'd') {
                // add left node
                TreeNode* new_node = new TreeNode{
                    static_cast<int32_t>(this->size),  // id
                    EinsumTree::node_t::leaf,          // node_type
                    nullptr,                           // parent
                    nullptr,                           // left_child
                    nullptr,                           // right_child
                    {},                                // notation
                    nullptr,                           // left_tensor
                    nullptr,                           // right_tensor
                    nullptr,                           // out_child
                    TensorOperation::prim_t::none,     // first touch
                    TensorOperation::prim_t::none,     // operation primitive
                    TensorOperation::prim_t::none,     // last touch
                    TensorOperation(),                 // op
                };

                if (this->leaf_ids.size() > 0) {
                    if (this->leaf_ids.back() == current->id) {
                        this->leaf_ids.pop_back();
                    }
                }

                current->left_child = new_node;
                current->node_type = EinsumTree::node_t::permutation;
                current->operation_primitive = TensorOperation::prim_t::copy;
                new_node->parent = current;
                current = new_node;
                stack.push_back('l');
                this->leaf_ids.push_back(static_cast<int32_t>(this->size));
                this->size++;
            } else if (stack.back() == 'r') {
                // add right node
                TreeNode* new_node = new TreeNode{
                    static_cast<int32_t>(this->size),  // id
                    EinsumTree::node_t::leaf,          // node_type
                    nullptr,                           // parent
                    nullptr,                           // left_child
                    nullptr,                           // right_child
                    {},                                // notation
                    nullptr,                           // left_tensor
                    nullptr,                           // right_tensor
                    nullptr,                           // out_child
                    TensorOperation::prim_t::none,     // first touch
                    TensorOperation::prim_t::none,     // operation primitive
                    TensorOperation::prim_t::none,     // last touch
                    TensorOperation(),                 // op
                };
                current->right_child = new_node;
                current->node_type = EinsumTree::node_t::contraction;
                current->operation_primitive = TensorOperation::prim_t::gemm;
                new_node->parent = current;
                current = new_node;
                // mark right node as "done" in stack
                stack.back() = 'd';
                this->leaf_ids.push_back(static_cast<int32_t>(this->size));
                this->size++;
            } else if (stack.back() == 'u') {
                // "stop" and write into node (afterwards continue upwards)
                stack.push_back('s');
            }

        } else if (character >= '0' && character <= '9') {
            if (stack.back() == 'l' || stack.back() == 'd' || stack.back() == 's') {
                // mark as currently "writing" into current node
                stack.push_back('w');
            }
            current->notation.push_back(static_cast<uint32_t>(character - '0'));
        } else if (character == ',') {
            if (stack.back() == 'l' || stack.back() == 'd' || stack.back() == 'u') {
                // go to right next
                if (stack.back() == 'u') {
                    // clean stack (don't go further up)
                    stack.pop_back();
                } else if (stack.back() == 'l' || stack.back() == 'd') {
                    // go to parent to reach right neighbor
                    current = current->parent;
                }
                stack.push_back('r');
            }
        } else if (character == ']') {
            if (stack.back() == 'w') {
                // if notation in current get rid off 'w' and 'l', 'd' or 's'
                stack.pop_back();
                stack.pop_back();
            } else if (stack.back() == 'u') {
                // if this is another bracket to be closed go further up in tree
                current = current->parent;
                stack.pop_back();
                stack.pop_back();
                stack.push_back('u');
            }
        } else if (character == '>') {
            // only go to parent if not already at parent
            if (stack.back() != 'u') {
                stack.push_back('u');
                if (current->parent != nullptr) {
                    current = current->parent;
                }
            }
        } else if (character >= 'a' && character <= 'z') {
            if (character == 'r') {
                if (stack.back() == 'w') {
                    current->first_touch = TensorOperation::prim_t::relu;
                } else {
                    current->last_touch = TensorOperation::prim_t::relu;
                }
            } else if (character == 'z') {
                if (stack.back() == 'w') {
                    current->first_touch = TensorOperation::prim_t::zero;
                } else {
                    current->last_touch = TensorOperation::prim_t::zero;
                }
            }
        }
    }

    // set tensors for each contraction
    identify();
}

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

void EinsumTree::optimize() {
    this->leaf_ids = {};
    if (this->root == nullptr) {
        std::cerr << "Einsum tree is empty, cannot optimize." << std::endl;
        return;
    }

    optimizeNode(this->root);
    identify();
}

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

void EinsumTree::optimizeNode(TreeNode* node) {
    if (node == nullptr) {
        std::cerr << "Node is null, cannot optimize." << std::endl;
        return;
    }
    if (node->node_type == node_t::leaf) {
        this->leaf_ids.push_back(node->id);
    } else if (node->node_type == node_t::permutation) {
        if (node->left_child == nullptr) {
            std::cerr << "Permutation node must have a left child." << std::endl;
            return;
        }

        // Recursively optimize left child
        optimizeNode(node->left_child);
    } else if (node->node_type == node_t::contraction) {
        if (node->left_child == nullptr || node->right_child == nullptr) {
            std::cerr << "Contraction node must have both left and right children." << std::endl;
            return;
        }

        // Check if the left and right children have to be swapped
        double current_score = getScore(node, TensorOperation::dim_t::m, false);
        double swap_score = getScore(node, TensorOperation::dim_t::n, true);
        if (swap_score < current_score) {
            swap(node);
        }

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
                TensorOperation::prim_t::none,    // first touch
                TensorOperation::prim_t::copy,    // operation primitive
                TensorOperation::prim_t::none,    // last touch
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
                TensorOperation::prim_t::none,    // first touch
                TensorOperation::prim_t::copy,    // operation primitive
                TensorOperation::prim_t::none,    // last touch
                TensorOperation(),                // op
            };
            insertPermutation(node, new_right_child, false);
        }

        // Recursively optimize left and right children
        optimizeNode(node->left_child);
        optimizeNode(node->right_child);
    }
}

void EinsumTree::identify() {
    identifyNode(this->root);
}

std::vector<uint32_t> EinsumTree::identifyNode(TreeNode* node) {
    std::vector<uint32_t> out_dims;
    for (uint32_t dim_id : node->notation) {
        out_dims.push_back(this->id_dims[dim_id]);
    }
    node->out_tensor = new Tensor(out_dims);
    int32_t size = 1;
    for (auto id : node->out_tensor->id) {
        if ((id.dim_t == static_cast<int>(TensorOperation::dim_t::m)) || (id.dim_t == static_cast<int>(TensorOperation::dim_t::n))) {
            size *= id.dim_sizes;
        }
    }

    float* output_f = new float[size];
    void* output = static_cast<void*>(output_f);
    if (node->node_type == node_t::permutation) {
        std::vector<uint32_t> child_dims = identifyNode(node->left_child);
        node->left_tensor = new Tensor(child_dims);
    } else if (node->node_type == node_t::contraction) {
        std::vector<uint32_t> left_dims = identifyNode(node->left_child);
        node->left_tensor = new Tensor(left_dims);

        std::vector<uint32_t> right_dims = identifyNode(node->right_child);
        node->right_tensor = new Tensor(right_dims);

        for (size_t i = 0; i < node->left_child->notation.size(); i++) {
            for (size_t j = 0; j < node->right_child->notation.size(); j++) {
                for (size_t k = 0; k < node->notation.size(); k++) {
                    bool is_contraction_dim = node->left_child->notation[i] == node->right_child->notation[j] &&
                                              node->left_child->notation[i] != node->notation[k];
                    bool is_m_dim = node->notation[k] != node->right_child->notation[j] &&
                                    node->notation[k] == node->left_child->notation[i];
                    bool is_n_dim = node->notation[k] == node->right_child->notation[j] &&
                                    node->notation[k] != node->left_child->notation[i];

                    if (is_contraction_dim) {
                        node->left_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::k);
                        node->right_tensor->id[j].dim_t = static_cast<int>(TensorOperation::dim_t::k);
                    } else if (is_m_dim) {
                        node->left_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::m);
                        node->out_tensor->id[k].dim_t = static_cast<int>(TensorOperation::dim_t::m);
                    } else if (is_n_dim) {
                        node->right_tensor->id[j].dim_t = static_cast<int>(TensorOperation::dim_t::n);
                        node->out_tensor->id[k].dim_t = static_cast<int>(TensorOperation::dim_t::n);
                    }
                }
            }
        }
    }

    return out_dims;
}

void EinsumTree::lower() {
    lowerNode(this->root);
}

TensorOperation::prim_t EinsumTree::lowerNode(TreeNode* node) {
    TensorOperation::prim_t node_op;
    if (node->node_type == node_t::leaf) {
        node_op = TensorOperation::prim_t::none;
    } else if (node->node_type == node_t::permutation) {
        TensorOperation::prim_t child_op = lowerNode(node->left_child);
    } else if (node->node_type == node_t::contraction) {
        this->bias_ids.push_back(node->id);
        TensorOperation::prim_t left_op = lowerNode(node->left_child);
        TensorOperation::prim_t right_op = lowerNode(node->right_child);

        TensorOperation::dtype_t dtype = TensorOperation::dtype_t::fp32;
        TensorOperation::prim_t prim_first_touch = node->first_touch;
        TensorOperation::prim_t prim_main = node->operation_primitive;
        TensorOperation::prim_t prim_last_touch = node->last_touch;

        std::vector<TensorOperation::dim_t> dim_types;
        std::vector<TensorOperation::exec_t> exec_types;
        std::vector<int64_t> dim_sizes;
        std::vector<int64_t> strides_in0;
        std::vector<int64_t> strides_in1;
        std::vector<int64_t> strides_out;
        std::unordered_set<uint32_t> added_dims;

        uint32_t dim_id = 0;

        for (uint32_t dim_size : this->id_dims) {
            TensorOperation::dim_t dim_type = TensorOperation::dim_t::undefined;
            int64_t stride_in0 = 0;
            int64_t stride_in1 = 0;
            int64_t stride_out = 0;

            for (size_t i = 0; i < node->left_child->notation.size(); i++) {
                if (dim_id == node->left_child->notation[i]) {
                    dim_type = static_cast<TensorOperation::dim_t>(node->left_tensor->id[i].dim_t);
                    stride_in0 = node->left_tensor->id[i].stride;
                    // break;
                }
                node->left_tensor->id[i].loop_id = node->left_child->notation[i];
            }

            for (size_t i = 0; i < node->right_child->notation.size(); i++) {
                if (dim_id == node->right_child->notation[i]) {
                    if (dim_type == TensorOperation::dim_t::undefined) {
                        dim_type = static_cast<TensorOperation::dim_t>(node->right_tensor->id[i].dim_t);
                    }
                    stride_in1 = node->right_tensor->id[i].stride;
                    // break;
                }
                node->right_tensor->id[i].loop_id = node->right_child->notation[i];
            }

            for (size_t i = 0; i < node->notation.size(); i++) {
                if (dim_id == node->notation[i]) {
                    stride_out = node->out_tensor->id[i].stride;
                    // break;
                }
                node->out_tensor->id[i].loop_id = node->notation[i];
            }

            if (dim_type != TensorOperation::dim_t::undefined) {
                dim_types.push_back(dim_type);
                exec_types.push_back(TensorOperation::exec_t::seq);
                dim_sizes.push_back(dim_size);
                strides_in0.push_back(stride_in0);
                Tensor::DimInfo dim_info{.dim_t = static_cast<int>(dim_type), .dim_sizes = static_cast<int>(dim_size), .stride = 0, .loop_id = static_cast<int>(dim_id)};
                if (stride_in0 == 0) {
                    node->left_tensor->id.push_back(dim_info);  // Set default stride if not set
                }
                strides_in1.push_back(stride_in1);
                if (stride_in1 == 0) {
                    node->right_tensor->id.push_back(dim_info);  // Set default stride if not set
                }
                strides_out.push_back(stride_out);
                if (stride_out == 0) {
                    node->out_tensor->id.push_back(dim_info);  // Set default stride if not set
                }
            }
            dim_id++;
        }

        Tensor* bias_tensor = nullptr;
        if (this->use_bias) {
            std::vector<u_int32_t> bias_dims;
            for (size_t i = 0; i < node->out_tensor->id.size(); i++) {
                if (node->out_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::n)) {
                    bias_dims.push_back(node->out_tensor->id[i].dim_sizes);
                } else if (node->out_tensor->id[i].dim_t == static_cast<int>(TensorOperation::dim_t::m)) {
                    bias_dims.push_back(1);
                } else {
                    bias_dims.push_back(0);
                }
            }

            bias_tensor = new Tensor(bias_dims);
        }

        std::span<TensorOperation::dim_t> dim_types_span(dim_types);
        std::span<TensorOperation::exec_t> exec_types_span(exec_types);
        std::span<int64_t> dim_sizes_span(dim_sizes);
        std::span<int64_t> strides_in0_span(strides_in0);
        std::span<int64_t> strides_in1_span(strides_in1);
        std::span<int64_t> strides_out_span(strides_out);

        TensorOperation::error_t result = node->op.setup(
            dtype,
            prim_first_touch,
            prim_main,
            prim_last_touch,
            dim_types_span,
            exec_types_span,
            dim_sizes_span,
            strides_in0_span,
            strides_in1_span,
            strides_out_span);

        if (result != TensorOperation::error_t::success) {
            std::cerr << "Setup failed for contraction operation" << std::endl;
        }
        node->op.optimize();
        node->op.compile();
    }
    return node_op;
}

void EinsumTree::execute(std::vector<void*> inputs, std::vector<void*> biases, void* output) {
    if (this->root == nullptr) {
        std::cerr << "Einsum tree is empty, cannot execute." << std::endl;
        return;
    }

    // Execute the root node and get the result
    void* result = executeNode(this->root, inputs, biases);

    if (result == nullptr) {
        std::cerr << "Execution failed, result is null." << std::endl;
        return;
    }

    // Copy the result to the output buffer
    // First calculate the total size of the output tensor
    int32_t size = 1;
    for (auto id : this->root->out_tensor->id) {
        if ((id.dim_t == static_cast<int>(TensorOperation::dim_t::m)) || (id.dim_t == static_cast<int>(TensorOperation::dim_t::n))) {
            size *= id.dim_sizes;
        }
    }

    // Copy the data
    memcpy(output, result, size * sizeof(float));
}

void* EinsumTree::executeNode(TreeNode* node, std::vector<void*> inputs, std::vector<void*> biases) {
    if (node == nullptr) {
        std::cerr << "Node is null, cannot execute." << std::endl;
        return nullptr;
    }

    if (node->node_type == EinsumTree::node_t::leaf) {
        // For leaf nodes, return the corresponding input data directly
        size_t index = 0;
        for (auto leaf_id : this->leaf_ids) {
            if (node->id == leaf_id) {
                return inputs[index];  // Return the input data directly
            }
            index++;
        }
        std::cerr << "Leaf node ID " << node->id << " not found in leaf_ids." << std::endl;
        return nullptr;
    }

    // For non-leaf nodes, we need to allocate output memory
    uint32_t out_size = 1;
    for (auto id : node->out_tensor->id) {
        out_size *= id.dim_sizes;  // Include ALL dimensions, not just m and n
    }

    float* output_f = new float[out_size]();  // Initialize to zero
    if (this->use_bias && node->node_type == EinsumTree::node_t::contraction) {
        // get correct bias for this node
        float* bias = nullptr;
        for (size_t i = 0; i < biases.size(); i++) {
            if (node->id == this->bias_ids[i]) {
                bias = (float*)biases[i];
                break;
            }
        }

        // get n and m dimensions of output tensor
        uint32_t n_size = 1;
        uint32_t m_size = 1;
        for (auto id : node->out_tensor->id) {
            if (id.dim_t == static_cast<int>(TensorOperation::dim_t::n)) {
                n_size *= id.dim_sizes;
            } else if (id.dim_t == static_cast<int>(TensorOperation::dim_t::m)) {
                m_size *= id.dim_sizes;
            }
        }

        // fill each column with the corresponding bias value
        for (size_t j = 0; j < n_size; j++) {
            std::fill(&output_f[j * m_size], &output_f[j * m_size + m_size], bias[j]);
        }
    }
    void* output = static_cast<void*>(output_f);

    if (node->node_type == EinsumTree::node_t::contraction) {
        // Execute left and right children
        void* left_output = executeNode(node->left_child, inputs, biases);
        void* right_output = executeNode(node->right_child, inputs, biases);

        if (left_output == nullptr || right_output == nullptr) {
            std::cerr << "Failed to execute child nodes." << std::endl;
            delete[] output_f;  // Clean up allocated memory
            return nullptr;
        }

        // Execute the tensor operation
        node->op.execute(left_output, right_output, output);
        if (node->left_child->node_type != EinsumTree::node_t::leaf) {
            // If the left child is not a leaf, we need to clean up the left output
            delete[] static_cast<float*>(left_output);
        }
        if (node->right_child->node_type != EinsumTree::node_t::leaf) {
            // If the right child is not a leaf, we need to clean up the right output
            delete[] static_cast<float*>(right_output);
        }
    } else if (node->node_type == EinsumTree::node_t::permutation) {
        // Execute child node
        void* child_output = executeNode(node->left_child, inputs, biases);

        if (child_output == nullptr) {
            std::cerr << "Failed to execute child node for permutation." << std::endl;
            delete[] output_f;  // Clean up allocated memory
            return nullptr;
        }

        // Execute the permutation operation
        node->op.execute(child_output, nullptr, output);
    } else {
        std::cerr << "Unsupported node type for execution: " << static_cast<int>(node->node_type) << std::endl;
        delete[] output_f;  // Clean up allocated memory
        return nullptr;
    }
    return output;
}

void EinsumTree::print() {
    if (this->root == nullptr) {
        std::cout << "Empty tree" << std::endl;
        return;
    }
    std::cout << "Notation; Node ID / First Touch | Operation Primitive | Last Touch" << std::endl;
    printNode(this->root, "", true);
    std::cout << "Leaf ID's: ";
    for (auto leaf_id : this->leaf_ids) {
        std::cout << leaf_id << " ";
    }
    std::cout << std::endl;
    std::cout << "Bias ID's: ";
    for (auto bias_id : this->bias_ids) {
        std::cout << bias_id << " ";
    }
    std::cout << std::endl;
}

void EinsumTree::printNode(TreeNode* node, const std::string& prefix, bool isLast) {
    if (node == nullptr) return;

    // Print current node
    std::cout << prefix;
    std::cout << (isLast ? "└─ " : "├─ ");

    // Print the vector notation
    for (size_t i = 0; i < node->notation.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << node->notation[i];
    }
    std::cout << "; " << node->id
              << " / " << static_cast<int>(node->first_touch)
              << " | " << static_cast<int>(node->operation_primitive)
              << " | " << static_cast<int>(node->last_touch) << std::endl;

    // Prepare prefix for children
    std::string childPrefix = prefix + (isLast ? "   " : "│  ");

    // Print children (if not a leaf)
    if (node->node_type != EinsumTree::node_t::leaf) {
        if (node->left_child != nullptr && node->right_child != nullptr) {
            // Both children exist
            printNode(node->left_child, childPrefix, false);
            printNode(node->right_child, childPrefix, true);
        } else if (node->left_child != nullptr) {
            // Only left child
            printNode(node->left_child, childPrefix, true);
        } else if (node->right_child != nullptr) {
            // Only right child
            printNode(node->right_child, childPrefix, true);
        }
    }
}

void EinsumTree::deleteNode(TreeNode* node) {
    if (node == nullptr) return;
    deleteNode(node->left_child);
    deleteNode(node->right_child);
    delete node->out_tensor;
    delete node->left_tensor;
    delete node->right_tensor;
    delete node;
}

void EinsumTree::delete_tree() {
    deleteNode(this->root);
    this->root = nullptr;
    this->size = 0;
    this->leaf_ids.clear();
    this->bias_ids.clear();
    this->id_dims.clear();
}