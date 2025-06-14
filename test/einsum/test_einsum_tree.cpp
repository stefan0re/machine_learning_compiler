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
    tree.identify();
    tree.print();
}

/*TEST_CASE("Einsum::Trees::EinsumTrees::parse test only binary", "[Einsum][Trees][EinsumTrees][parse]") {
    std::string str_repr = "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]";
    EinsumTree tree = EinsumTree(str_repr, {100, 72, 128, 128, 3, 71, 305, 32, 3});
    tree.print();
    tree.lower();
}

TEST_CASE("Einsum::Trees::EinsumTrees::parse test with unary", "[Einsum][Trees][EinsumTrees][parse]") {
    std::string str_repr = "[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]";
    EinsumTree tree = EinsumTree(str_repr, {60, 60, 20, 20, 8, 8, 8, 8, 8, 8});
    tree.print();
    tree.lower();
}*/