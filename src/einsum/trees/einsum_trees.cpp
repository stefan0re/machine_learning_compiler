#include "./einsum_trees.h"

#include <iostream>
#include <unordered_set>
#include <vector>

using namespace einsum::trees;
using namespace einsum::backend;

std::vector<int64_t> get_stride(std::vector<uint32_t> dim_ids,
                                std::vector<TensorOperation::dim_t> dim_types,
                                std::vector<int64_t> dim_sizes,
                                std::vector<TensorOperation::exec_t> exec_types,
                                TensorOperation::dim_t dim_type_0,
                                TensorOperation::dim_t dim_type_1) {
    std::vector<uint32_t> local_dim_ids;
    std::vector<TensorOperation::dim_t> local_dim_types;
    std::vector<uint32_t> local_dim_sizes;
    std::vector<TensorOperation::exec_t> local_exec_types;
    for (int i = 0; i < dim_types.size(); i++) {
        if (dim_types[i] == dim_type_0 || dim_types[i] == dim_type_1) {
            local_dim_ids.push_back(dim_ids[i]);
            local_dim_sizes.push_back(dim_sizes[i]);
            local_dim_types.push_back(dim_types[i]);
            local_exec_types.push_back(exec_types[i]);
        }
    }

    size_t count = 0;
    std::vector<int64_t> strides;
    for (auto dim : dim_ids) {
        strides.push_back(0);
        for (int i = 0; i < local_dim_ids.size(); i++) {
            if (dim == local_dim_ids[i]) {
                int64_t stride = 1;
                for (int j = i + 1; j < local_dim_ids.size(); j++) {
                    if (local_exec_types[i] == local_exec_types[j]) {
                        stride *= local_dim_sizes[j];
                    }
                }
                strides[count] = stride;
            }
        }
        count++;
    }
    return strides;
}

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

OpSteps EinsumTree::lower() {
    OpSteps steps = {{}, {}};
    OpSteps::OpStep lastStep = lowerNode(this->root, steps);
    return steps;
}

OpSteps::OpStep EinsumTree::lowerNode(TreeNode* node, OpSteps& lowered) {
    OpSteps::OpStep step;
    std::cout << "ID: " << node->id << " Type: " << static_cast<uint32_t>(node->node_type) << std::endl;
    if (node->node_type == node_t::leaf) {
        step.in_ten_left = -1;  // unused
        step.in_ten_left_notation = {};
        step.in_ten_right = -1;  // unused
        step.in_ten_right_notation = {};
        step.out_ten = node->id;  // output is leaf node
        step.out_ten_notation = node->notation;
    } else if (node->node_type == node_t::permutation) {
        OpSteps::OpStep child_step = lowerNode(node->left_child, lowered);
    } else if (node->node_type == node_t::contraction) {
        OpSteps::OpStep left_step = lowerNode(node->left_child, lowered);
        OpSteps::OpStep right_step = lowerNode(node->right_child, lowered);

        step.in_ten_left = left_step.out_ten;
        step.in_ten_left_notation = left_step.out_ten_notation;
        step.in_ten_right = right_step.out_ten;
        step.in_ten_right_notation = right_step.out_ten_notation;
        step.out_ten = node->id;  // output is current node
        step.out_ten_notation = node->notation;

        lowered.tensor_order.push_back(step);

        TensorOperation op;

        TensorOperation::dtype_t dtype = TensorOperation::dtype_t::fp32;
        TensorOperation::prim_t prim_first_touch = TensorOperation::prim_t::none;
        TensorOperation::prim_t prim_main = TensorOperation::prim_t::gemm;
        TensorOperation::prim_t prim_last_touch = TensorOperation::prim_t::none;

        /*for (int i = 0; i < dim_ids.size(); i++) {
    std::cout << "ID: " << dim_ids[i] << ", Dim Type: " << static_cast<uint32_t>(dim_types[i]) << ", Exec Type: " << static_cast<uint32_t>(exec_types[i]) << ", Dim Size: " << static_cast<uint32_t>(dim_sizes[i]) << ", Left Stride: " << static_cast<uint32_t>(strides_in0[i]) << ", Right Stride: " << static_cast<uint32_t>(strides_in1[i]) << ", Out Stride: " << static_cast<uint32_t>(strides_out[i]) << std::endl;
}

TensorOperation::error_t result = op.setup(
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
    // Fehlerbehandlung
    std::cerr << "Setup failed for contraction operation" << std::endl;
}*/

        // lowered.step_list.push_back(op);
    } else {
        return {0, {}, 0, {}, 0, {}};  // unknown type
    }
    return step;
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