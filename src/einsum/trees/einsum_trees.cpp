#include "./einsum_trees.h"

#include <iostream>
#include <vector>

einsum::trees::EinsumTree::EinsumTree(std::string str_repr, std::vector<uint32_t> id_dims) {
    this->id_dims = id_dims;

    std::vector<char> stack = {'l'};

    this->root = new TreeNode{
        {},       // notation
        false,    // is_binary
        true,     // is_leaf
        nullptr,  // parent
        nullptr,  // left_child
        nullptr   // right_child
    };

    TreeNode* current = this->root;

    for (char character : str_repr) {
        if (character == '[') {
            if (stack.back() == 'l' || stack.back() == 'd') {
                // add left node
                this->size++;
                TreeNode* new_node = new TreeNode{
                    {},       // notation
                    false,    // is_binary
                    true,     // is_leaf
                    nullptr,  // parent
                    nullptr,  // left_child
                    nullptr   // right_child
                };
                current->left_child = new_node;
                current->is_leaf = false;
                new_node->parent = current;
                current = new_node;
                stack.push_back('l');
            } else if (stack.back() == 'r') {
                // add right node
                this->size++;
                TreeNode* new_node = new TreeNode{
                    {},       // notation
                    false,    // is_binary
                    true,     // is_leaf
                    nullptr,  // parent
                    nullptr,  // left_child
                    nullptr   // right_child
                };
                current->right_child = new_node;
                current->is_leaf = false;
                new_node->parent = current;
                current = new_node;
                // mark right node as "done" in stack
                stack.back() = 'd';
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

void einsum::trees::EinsumTree::print() {
    if (root == nullptr) {
        std::cout << "Empty tree" << std::endl;
        return;
    }

    printNode(root, "", true);
}

void einsum::trees::EinsumTree::printNode(einsum::trees::EinsumTree::TreeNode* node, const std::string& prefix, bool isLast) {
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
    if (!node->is_leaf) {
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