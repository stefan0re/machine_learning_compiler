#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../../src/einsum/trees/einsum_trees.h"

using namespace einsum::trees;

TEST_CASE("Einsum::Trees::EinsumTrees::simple binary operation", "[Einsum][Trees][EinsumTrees]") {
    std::string str_repr = "[0,1],[1,2]->[0,2]";
    EinsumTree tree = EinsumTree(str_repr, {10, 20, 30});
    tree.optimize();
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
    // tree.execute({input1, input2}, output);
    for (int i = 0; i < 10 * 30; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    delete[] input1;
    delete[] input2;
    delete[] output;
}
/*
TEST_CASE("Einsum::Trees::EinsumTrees::parse test only binary", "[Einsum][Trees][EinsumTrees][parse]") {
    std::string str_repr = "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]";
    EinsumTree tree = EinsumTree(str_repr, {100, 72, 128, 128, 3, 71, 305, 32, 3});
    tree.print();
    tree.optimize();
    tree.print();
    tree.lower();
}

TEST_CASE("Einsum::Trees::EinsumTrees::parse test with unary", "[Einsum][Trees][EinsumTrees][parse]") {
    std::string str_repr = "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]";
    EinsumTree tree = EinsumTree(str_repr, {60, 60, 20, 20, 8, 8, 8, 8, 8, 8});
    tree.print();
    tree.optimize();
    tree.print();
    tree.lower();
}

TEST_CASE("Einsum::Trees::EinsumTrees::optimize 1", "[Einsum][Trees][EinsumTrees][optimize]") {
    std::string str_repr = "[[7,3,8],[8,4]->[7,3,4]],[[0,5],[[5,1,6],[6,2,7]->[5,1,2,7]]->[0,1,2,7]]->[0,1,2,3,4]";
    EinsumTree tree = EinsumTree(str_repr, {100, 72, 128, 128, 3, 71, 305, 32, 3});
    tree.print();
    tree.optimize();
    tree.print();
    tree.lower();
}

TEST_CASE("Einsum::Trees::EinsumTrees::optimize 2", "[Einsum][Trees][EinsumTrees][optimize]") {
    std::string str_repr = "[1,4,7,8],[[0,4,5,6],[[2,5,7,9],[3,6,8,9]->[2,5,7,3,6,8]]->[0,4,2,7,3,8]]->[0,1,2,3]";
    EinsumTree tree = EinsumTree(str_repr, {60, 60, 20, 20, 8, 8, 8, 8, 8, 8});
    tree.print();
    tree.optimize();
    tree.print();
    tree.lower();
}

TEST_CASE("Einsum::Trees::EinsumTrees::optimize 3", "[Einsum][Trees][EinsumTrees][optimize]") {
    std::string str_repr = "[[2,7,3],[3,8,4]->[2,7,8,4]],[[4,9,0],[[0,5,1],[1,6,2]->[0,5,6,2]]->[4,9,5,6,2]]->[5,6,7,8,9]";
    EinsumTree tree = EinsumTree(str_repr, {40, 40, 40, 40, 40, 25, 25, 25, 25, 25});
    tree.print();
    tree.optimize();
    tree.print();
    tree.lower();
}
    */