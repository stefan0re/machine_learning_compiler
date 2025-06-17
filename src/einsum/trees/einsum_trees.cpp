#include "./einsum_trees.h"

#include <iostream>
#include <unordered_set>
#include <vector>

using namespace einsum::trees;
using namespace einsum::backend;

EinsumTree::EinsumTree(std::string str_repr, std::vector<uint32_t> id_dims) {
    this->id_dims = id_dims;

    std::vector<char> stack = {'l'};

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
        TensorOperation(),                 // op
    };
    this->leaf_ids.push_back(static_cast<int32_t>(this->size));
    this->size++;

    TreeNode* current = this->root;

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
                    TensorOperation(),                 // op
                };

                if (this->leaf_ids.size() > 0) {
                    if (this->leaf_ids.back() == current->id) {
                        this->leaf_ids.pop_back();
                    }
                }

                current->left_child = new_node;
                current->node_type = EinsumTree::node_t::permutation;
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
                    TensorOperation(),                 // op
                };
                current->right_child = new_node;
                current->node_type = EinsumTree::node_t::contraction;
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
            if (stack.back() == 'l' || stack.back() == 'u') {
                // go to right next
                if (stack.back() == 'u') {
                    // clean stack (don't go further up)
                    stack.pop_back();
                } else if (stack.back() == 'l') {
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
        }
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

    if (node->node_type == node_t::permutation) {
        std::cout << "Permutation Tensor" << std::endl;
        node->out_tensor->info();

        std::vector<uint32_t> child_dims = identifyNode(node->left_child);
        node->left_tensor = new Tensor(child_dims);
        std::cout << "Child Tensor" << std::endl;
        node->left_tensor->info();
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
                        std::cout << "Contraction dim: " << node->left_child->notation[i] << std::endl;
                    } else if (is_m_dim) {
                        node->left_tensor->id[i].dim_t = static_cast<int>(TensorOperation::dim_t::m);
                        node->out_tensor->id[k].dim_t = static_cast<int>(TensorOperation::dim_t::m);
                        std::cout << "m dim: " << node->notation[k] << std::endl;
                    } else if (is_n_dim) {
                        node->right_tensor->id[j].dim_t = static_cast<int>(TensorOperation::dim_t::n);
                        node->out_tensor->id[k].dim_t = static_cast<int>(TensorOperation::dim_t::n);
                        std::cout << "n dim: " << node->notation[k] << std::endl;
                    }
                }
            }
        }

        std::cout << "Contraction Tensor" << std::endl;
        node->out_tensor->info();
        std::cout << "Left Tensor" << std::endl;
        node->left_tensor->info();
        std::cout << "Right Tensor" << std::endl;
        node->right_tensor->info();
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
        TensorOperation::prim_t left_op = lowerNode(node->left_child);
        TensorOperation::prim_t right_op = lowerNode(node->right_child);

        TensorOperation::dtype_t dtype = TensorOperation::dtype_t::fp32;
        TensorOperation::prim_t prim_first_touch = TensorOperation::prim_t::none;
        TensorOperation::prim_t prim_main = TensorOperation::prim_t::gemm;
        TensorOperation::prim_t prim_last_touch = TensorOperation::prim_t::none;

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
                    break;
                }
            }

            for (size_t i = 0; i < node->right_child->notation.size(); i++) {
                if (dim_id == node->right_child->notation[i]) {
                    if (dim_type == TensorOperation::dim_t::undefined) {
                        dim_type = static_cast<TensorOperation::dim_t>(node->right_tensor->id[i].dim_t);
                    }
                    stride_in1 = node->right_tensor->id[i].stride;
                    break;
                }
            }

            for (size_t i = 0; i < node->notation.size(); i++) {
                if (dim_id == node->notation[i]) {
                    stride_out = node->out_tensor->id[i].stride;
                    break;
                }
            }

            if (dim_type != TensorOperation::dim_t::undefined) {
                dim_types.push_back(dim_type);
                exec_types.push_back(TensorOperation::exec_t::seq);
                dim_sizes.push_back(dim_size);
                strides_in0.push_back(stride_in0);
                strides_in1.push_back(stride_in1);
                strides_out.push_back(stride_out);
            }
            dim_id++;
        }

        TensorOperation::error_t result = node->op.setup(
            dtype,
            prim_first_touch,
            prim_main,
            prim_last_touch,
            std::span<const TensorOperation::dim_t>(dim_types),
            std::span<const TensorOperation::exec_t>(exec_types),
            std::span<const int64_t>(dim_sizes),
            std::span<const int64_t>(strides_in0),
            std::span<const int64_t>(strides_in1),
            std::span<const int64_t>(strides_out));

        if (result != TensorOperation::error_t::success) {
            std::cerr << "Setup failed for contraction operation" << std::endl;
        }

        node->op.optimize();
        node->op.compile();
    }
    return node_op;
}

void EinsumTree::execute(std::vector<void*> inputs, void* output) {
    if (this->root == nullptr) {
        std::cerr << "Einsum tree is empty, cannot execute." << std::endl;
        return;
    }

    // Execute the root node
    output = executeNode(this->root, inputs);
}

void* EinsumTree::executeNode(TreeNode* node, std::vector<void*> inputs) {
    if (node == nullptr) {
        std::cerr << "Node is null, cannot execute." << std::endl;
        return nullptr;
    }

    void* output = nullptr;

    if (node->node_type == EinsumTree::node_t::leaf) {
        size_t index = 0;
        for (auto leaf_id : this->leaf_ids) {
            if (node->id == leaf_id) {
                output = static_cast<void*>(inputs[index]);
                break;
            }
            index++;
        }
    } else if (node->node_type == EinsumTree::node_t::contraction) {
        // Execute left and right children
        void* left_output = executeNode(node->left_child, inputs);
        void* right_output = executeNode(node->right_child, inputs);
        if (left_output == nullptr || right_output == nullptr) {
            std::cerr << "Failed to execute child nodes." << std::endl;
        }
        node->op.execute(left_output, right_output, output);
    } else {
        std::cerr << "Unsupported node type for execution." << std::endl;
    }

    // Perform the operation defined in the node
    return output;
}

void EinsumTree::print() {
    if (this->root == nullptr) {
        std::cout << "Empty tree" << std::endl;
        return;
    }

    printNode(this->root, "", true);
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
    std::cout << std::endl;

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