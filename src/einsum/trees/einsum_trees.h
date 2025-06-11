#ifndef EINSUM_TREES_EINSUM_TREE_H
#define EINSUM_TREES_EINSUM_TREE_H

#include <string>

namespace einsum {
    namespace backend {
        class TensorOperation;
    }
}  // namespace einsum

class einsum::trees::EinsumTree {
   private:
    struct TreeNode {
        std::string notation;
        bool is_binary;
        TreeNode* parent;
        TreeNode* left_child;
        TreeNode* right_child;
    };

   public:
    EinsumTree(string str_repr);
};

#endif