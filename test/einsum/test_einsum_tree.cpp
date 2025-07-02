#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "einsum_trees.h"
#include "test_utils.h"

using namespace einsum::trees;

TEST_CASE("Einsum::Trees::EinsumTrees::simple binary operation", "[Einsum][Trees][EinsumTrees]") {
    std::cout << "########## Einsum tree test case 1 ##########" << std::endl;
    std::string str_repr = "[0,1,z],[1,2]r->[0,2]r";
    EinsumTree tree = EinsumTree(str_repr, {10, 20, 30});
    tree.optimize();
    tree.print();
    tree.lower();
    float* input1 = new float[10 * 20];
    float* input2 = new float[20 * 30];
    float* output = new float[10 * 30];
    for (int i = 0; i < 10 * 20; ++i) {
        input1[i] = static_cast<float>(rand() % 100);
    }

    for (int i = 0; i < 20 * 30; ++i) {
        input2[i] = static_cast<float>(rand() % 100);
    }
    tree.execute({input1, input2}, {}, output);
    for (int i = 0; i < 10 * 30; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    tree.delete_tree();
}

/*TEST_CASE("Einsum::Trees::EinsumTrees::parse test only binary", "[Einsum][Trees][EinsumTrees][parse]") {
    std::cout << "########### Einsum tree test case 2 ##########" << std::endl;
    std::string str_repr = "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]";
    EinsumTree tree = EinsumTree(str_repr, {100, 72, 128, 128, 3, 71, 305, 32, 3});
    tree.optimize();
    tree.print();
    tree.lower();
    tree.delete_tree();
}*/

TEST_CASE("Einsum::Trees::EinsumTrees::parse test with unary", "[Einsum][Trees][EinsumTrees][parse]") {
    std::cout << "########### Einsum tree test case 3 ##########" << std::endl;
    std::string str_repr = "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]";
    EinsumTree tree = EinsumTree(str_repr, {60, 60, 20, 20, 8, 8, 8, 8, 8, 8});
    tree.optimize();
    tree.print();
    tree.lower();
    tree.delete_tree();
}

TEST_CASE("Einsum::Trees::EinsumTrees::optimize 1", "[Einsum][Trees][EinsumTrees][optimize]") {
    std::cout << "########### Einsum tree test case 4 ##########" << std::endl;
    std::string str_repr = "[[7,3,8],[8,4]->[7,3,4]],[[0,5],[[5,1,6],[6,2,7]->[5,1,2,7]]->[0,1,2,7]]->[0,1,2,3,4]";
    EinsumTree tree = EinsumTree(str_repr, {100, 72, 128, 128, 3, 71, 305, 32, 3});
    tree.optimize();
    tree.print();
    tree.lower();
    tree.delete_tree();
}

TEST_CASE("Einsum::Trees::EinsumTrees::optimize 2", "[Einsum][Trees][EinsumTrees][optimize]") {
    std::cout << "########### Einsum tree test case 5 ##########" << std::endl;
    std::string str_repr = "[1,4,7,8],[[0,4,5,6],[[2,5,7,9],[3,6,8,9]->[2,5,7,3,6,8]]->[0,4,2,7,3,8]]->[0,1,2,3]";
    EinsumTree tree = EinsumTree(str_repr, {60, 60, 20, 20, 8, 8, 8, 8, 8, 8});
    tree.optimize();
    tree.print();
    tree.lower();
    tree.delete_tree();
}

TEST_CASE("Einsum::Trees::EinsumTrees::optimize 3", "[Einsum][Trees][EinsumTrees][optimize]") {
    std::cout << "######## Einsum tree test case 6 ##########" << std::endl;
    std::string str_repr = "[[2,7,3],[3,8,4]->[2,7,8,4]],[[4,9,0],[[0,5,1],[1,6,2]->[0,5,6,2]]->[4,9,5,6,2]]->[5,6,7,8,9]";
    EinsumTree tree = EinsumTree(str_repr, {40, 40, 40, 40, 40, 25, 25, 25, 25, 25});
    tree.optimize();
    tree.print();
    tree.lower();
    tree.delete_tree();
}

TEST_CASE("Einsum::Trees::EinsumTrees::small model and bias", "[Einsum][Trees][EinsumTrees][small_model]") {
    std::cout << "######## Einsum tree test case 7 ##########" << std::endl;
    std::string str_repr = "[1,0],[2,1]->[2,0]";
    EinsumTree tree = EinsumTree(str_repr, {4, 4, 4}, true);
    tree.optimize();
    tree.lower();
    tree.print();

    float* input1 = new float[4 * 4];
    float* input2 = new float[4 * 4];
    float* bias1 = new float[4 * 1];
    float* output = new float[4 * 4];
    test::matmul::generate_matrix(4, 4, input1, true, true);
    test::matmul::generate_matrix(4, 4, input2, true, true);
    test::matmul::generate_matrix(4, 1, bias1, false, true);

    test::matmul::print_matrix(4, 4, input1);
    test::matmul::print_matrix(4, 4, input2);
    test::matmul::print_matrix(4, 1, bias1);

    tree.execute({input1, input2}, {bias1}, output);

    float* output_ref = new float[4 * 4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int index = j * 4 + i;
            output_ref[index] = input1[index] * input2[index] + bias1[j];
        }
    }

    bool is_correct = test::matmul::compare_matrix(4, 4, output, output_ref);
    REQUIRE(is_correct);

    delete[] bias1;
    delete[] output;
    delete[] output_ref;
    tree.delete_tree();
    std::cout << "Test completed successfully." << std::endl;
}

TEST_CASE("Einsum::Trees::EinsumTrees::small tree", "[Einsum][Trees][EinsumTrees][small_tree]") {
    std::cout << "######## Einsum tree test case 8 ##########" << std::endl;
    std::string str_repr = "[0, 1],[[2, 3],[2, 1]->[3, 1]]->[0, 3]";
    EinsumTree tree = EinsumTree(str_repr, {4, 4, 4, 4});
    tree.print();

    tree.optimize();
    tree.print();
    tree.delete_tree();
}
