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

    /**
     * @brief Prints the structure of a Einsum tree node.
     *
     * @param node current node in the tree to be printed.
     * @param prefix String prefix for formatting the output.
     * @param isLast Boolean indicating if the node is the last child in its parent.
     */
    void printNode(TreeNode* node, const std::string& prefix, bool isLast);
    /**
     * @brief Identify the dimensions of the tensors for a node.
     *
     * @param node current node in the tree for which tensor are identified.
     * @return std::vector<uint32_t> list of dimensions for out tensor.
     */
    std::vector<uint32_t> identifyNode(TreeNode* node);
    void ::OpStep lowerNode(TreeNode* node, OpSteps& lowered);

   public:
    /**
     * @brief Construct a new Einsum Tree object and parses string representation.
     * The string representation should be in the format:
     * "[left_tensor_notation],[right_tensor_notation]->[output_tensor_notation]"
     * Example: "[0,1],[1,2]->[0,2]"
     *
     * @param str_repr String representation of the einsum operation.
     * @param id_dims Vector of dimensions for each tensor ID in the einsum operation.
     */
    EinsumTree(std::string str_repr, std::vector<uint32_t> id_dims);
    /**
     * @brief Identify the dimensions of the tensors in the tree.
     */
    void identify();
    void lower();
    /**
     * @brief Prints the structure of the Einsum tree.
     */
    void print();
};

#endif