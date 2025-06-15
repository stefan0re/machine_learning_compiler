#ifndef EINSUM_TREES_EINSUM_TREE_H
#define EINSUM_TREES_EINSUM_TREE_H

#include <cstdint>
#include <string>
#include <vector>

#include "../../tensor/tensor.h"
#include "../backend/TensorOperation.h"

namespace einsum {
    namespace trees {
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

        class EinsumTree;
    }  // namespace trees
}  // namespace einsum

class einsum::trees::EinsumTree {
   private:
    enum class node_t : uint32_t {
        leaf = 0,
        permutation = 1,
        contraction = 2,
    };

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
    };

    TreeNode* root = nullptr;
    uint32_t size = 0;
    std::vector<uint32_t> id_dims = {};
    std::vector<int32_t> leaf_ids = {};

    void printNode(TreeNode* node, const std::string& prefix, bool isLast);
    std::vector<uint32_t> identifyNode(TreeNode* node);
    OpSteps::OpStep lowerNode(TreeNode* node, OpSteps& lowered);

   public:
    EinsumTree(std::string str_repr, std::vector<uint32_t> id_dims);
    void identify();
    OpSteps lower();

    void print();
};

#endif