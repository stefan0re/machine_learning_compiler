#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../../src/einsum/trees/einsum_trees.h"
#include "../../src/mini_jit/include/gemm_ref.h"

using namespace einsum::trees;

TEST_CASE("Einsum::Trees::EinsumTrees::simple binary operation", "[Einsum][Trees][EinsumTrees]") {
    std::string str_repr = "[1,0],[2,1]->[2,0]";
    EinsumTree tree = EinsumTree(str_repr, {7, 46, 88});
    tree.optimize();
    tree.print();
    tree.lower();
    float* in0 = new float[7 * 46];
    float* in1 = new float[46 * 88];
    float* out = new float[7 * 88];
    float* out_ref = new float[7 * 88];

    srand48(time(NULL));
    for (size_t i = 0; i < 7 * 46; i++) {
        in0[i] = (float)drand48();
    }
    for (size_t i = 0; i < 46 * 88; i++) {
        in1[i] = (float)drand48();
    }
    for (size_t i = 0; i < 7 * 88; i++) {
        out[i] = 0.0f;
        out_ref[i] = 0.0f;
    }
    tree.execute({in0, in1}, {}, out);

    gemm_ref(in0, in1, out_ref, 7, 88, 46, 7, 46, 7);

    double error = 0;
    for (size_t i = 0; i < 7 * 88; i++) {
        error += std::abs(out[i] - out_ref[i]);
    }
    std::cout << "Error: " << error << std::endl;
    REQUIRE(error < 1e-5);

    tree.delete_tree();
}
