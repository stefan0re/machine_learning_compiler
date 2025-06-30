#ifndef EINSUM_TREES_EINSUM_TREE_H
#define EINSUM_TREES_EINSUM_TREE_H

#include <cstdint>
#include <string>
#include <vector>

#include "../../tensor/tensor.h"
#include "../backend/TensorOperation.h"

namespace einsum {
    namespace trees {
        class EinsumTree;
    }  // namespace trees
}  // namespace einsum

using namespace einsum::backend;

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

        TensorOperation::prim_t first_touch;
        TensorOperation::prim_t operation_primitive;
        TensorOperation::prim_t last_touch;

        TensorOperation op;
    };

    TreeNode* root = nullptr;
    uint32_t size = 0;
    bool use_bias = false;

    std::vector<uint32_t> id_dims = {};
    std::vector<int32_t> leaf_ids = {};
    std::vector<uint32_t> bias_ids = {};
    std::vector<void*> allocated_memory = {};  // Track allocated memory for cleanup

    /**
     * @brief Prints the structure of a Einsum tree node.
     *
     * @param node current node in the tree to be printed.
     * @param prefix String prefix for formatting the output.
     * @param isLast Boolean indicating if the node is the last child in its parent.
     */
    void printNode(TreeNode* node, const std::string& prefix, bool isLast);
    /**
     * @brief Identify the dimensions of the tensors in the tree.
     */
    void identify();
    /**
     * @brief Identify the dimensions of the tensors for a node.
     *
     * @param node current node in the tree for which tensor are identified.
     * @return std::vector<uint32_t> list of dimensions for out tensor.
     */
    std::vector<uint32_t> identifyNode(TreeNode* node);
    /**
     * @brief Lowers the Einsum tree nodes to hold tensor operations.
     *
     * @param node current node in the tree to be lowered.
     * @return TensorOperation::prim_t The lowered tensor operation primitive type for the node.
     */

    TensorOperation::prim_t lowerNode(TreeNode* node);
    /**
     * @brief Executes the Einsum tree nodes recursively.
     *
     * @param node current node in the tree to be executed.
     * @param inputs Vector of input tensors for the execution.
     * @return void* Pointer to the output tensor after execution.
     */
    void* executeNode(TreeNode* node, std::vector<void*> inputs, std::vector<void*> biases);
    /**
     * @brief Swaps the left and right children of a node if the the parent is contraction.
     *
     * @param parent Pointer to the parent node whose children are to be swapped.
     * @return void* Pointer to the output tensor after swapping.
     */
    void swap(TreeNode* parent);
    /**
     * @brief Inserts a new child permutation node into the tree.
     *
     * @param parent Pointer to the parent node where the new child will be inserted.
     * @param new_child Pointer to the new child node to be inserted.
     * @param is_left Boolean indicating if the new child should be inserted as a left child.
     */
    void insertPermutation(TreeNode* parent, TreeNode* new_child, bool is_left);
    /**
     * @brief Calculates the score for a node based on its dimensions and whether to swap children.
     *
     * @param node  Pointer to the current node in the tree.
     * @param dim_type The dimension type to consider for scoring (e.g., m, n, k).
     * @param swap Boolean indicatiing whether to look at the swapped dimensions.
     * @return double The score for the node based on its dimensions.
     */
    double getScore(TreeNode* node, TensorOperation::dim_t dim_type, bool swap);
    /**
     * @brief Optimizes the Einsum tree nodes by adding permutation nodes and swapping children if necessary.
     *
     * @param node Pointer to the current node in the tree to be optimized.
     */
    void optimizeNode(TreeNode* node);

   public:
    /**
     * @brief Construct a new Einsum Tree object and parses string representation.
     * The string representation should be in the format:
     * "[left_tensor_notation],[right_tensor_notation]->[output_tensor_notation]"
     * Example: "[0,1],[1,2]->[0,2]"
     *
     * @param str_repr String representation of the einsum operation.
     * @param id_dims Vector of dimensions for each tensor ID in the einsum operation.
     * @param use_bias Boolean indicating whether to use a bias tensor in the operation.
     */
    EinsumTree(std::string str_repr, std::vector<uint32_t> id_dims, bool use_bias = false);
    /**
     * @brief Lowers the Einsum tree nodes for each to hold a tensor operations.
     */
    void lower();
    /**
     * @brief Optimizes the Einsum tree for efficient execution.
     * This includes adding permutation nodes and swapping children if necessary.
     */
    void optimize();
    /**
     * @brief Executes the Einsum tree with the provided input tensors.
     *
     * @param inputs Vector of input tensors to be used in the execution.
     * @return void* Pointer to the output tensor after execution.
     */
    void execute(std::vector<void*> inputs, std::vector<void*> biases, void* output);
    /**
     * @brief Prints the structure of the Einsum tree.
     */
    void print();
    /**
     * @brief Cleans up allocated memory from intermediate computations.
     */
    void cleanup();
};

#endif