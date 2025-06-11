#ifndef EINSUM_TREES_EINSUM_TREE_H
#define EINSUM_TREES_EINSUM_TREE_H

#include <cstdint>
#include <string>
#include <vector>

namespace einsum {
    namespace trees {
        class EinsumTree;
    }
}  // namespace einsum

class einsum::trees::EinsumTree {
   private:
    struct TreeNode {
        std::vector<uint32_t> notation;
        bool is_leaf;
        TreeNode* parent;
        TreeNode* left_child;
        TreeNode* right_child;
    };

    TreeNode* root = nullptr;
    std::vector<uint32_t> id_dims = {};
    uint32_t size = 0;
    void printNode(TreeNode* node, const std::string& prefix, bool isLast);

   public:
    EinsumTree(std::string str_repr, std::vector<uint32_t> id_dims);
    void execute();

    void print();
};

#endif