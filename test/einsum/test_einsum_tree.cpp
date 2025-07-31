#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../../src/einsum/trees/einsum_trees.h"
#include "../../src/mini_jit/include/gemm_ref.h"
#include "../test_utils/test_utils.h"

using namespace einsum::trees;

TEST_CASE("Einsum::Trees::EinsumTrees::simple binary operation", "[Einsum][Trees][EinsumTrees]") {
    std::string str_repr = "[1,0],[2,1]->[2,0]";
    EinsumTree tree = EinsumTree(str_repr, {7, 46, 88});
    tree.optimize();
    tree.lower();
    tree.print();
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

TEST_CASE("Einsum::Trees::EinsumTrees::simple bias op", "[Einsum][Trees][EinsumTrees]") {
    std::string str_repr = "[1,0],[2,1]->[2,0]";
    // m=5, n=6, k=4
    EinsumTree tree = EinsumTree(str_repr, {5, 4, 6}, true);
    tree.optimize();
    tree.lower();
    tree.print();
    float* in0 = new float[5 * 4];
    float* in1 = new float[4 * 6];
    float* bias = new float[6];
    float* out = new float[5 * 6];
    float* out_ref = new float[5 * 6];

    srand48(time(NULL));
    for (size_t i = 0; i < 5 * 4; i++) {
        in0[i] = (float)drand48();
    }
    for (size_t i = 0; i < 4 * 6; i++) {
        in1[i] = (float)drand48();
    }
    for (size_t i = 0; i < 6; i++) {
        bias[i] = (float)(i + 1);
    }
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 6; j++) {
            out[i] = 0.0f;
            out_ref[j * 5 + i] = bias[j];
        }
    }

    tree.execute({in0, in1}, {bias}, out);

    gemm_ref(in0, in1, out_ref, 5, 6, 4, 5, 4, 5);

    double error = 0;
    for (size_t i = 0; i < 5 * 6; i++) {
        error += std::abs(out[i] - out_ref[i]);
    }
    std::cout << "Error: " << error << std::endl;
    REQUIRE(error < 1e-5);

    tree.delete_tree();
}

TEST_CASE("Einsum::Trees::EinsumTrees::large bias tree", "[Einsum][Trees][EinsumTrees]") {
    std::string str_repr = "[[[1,0],[2,1]->[2,0]],[3,2]->[3,0]],[4,3]->[4,0]";
    // m=5, n=6, k=4
    EinsumTree tree = EinsumTree(str_repr, {5, 4, 6, 7, 8}, true);
    tree.optimize();
    tree.lower();
    tree.print();
    float* in0 = new float[5 * 4];
    float* in1 = new float[4 * 6];
    float* bias0 = new float[6];

    float* in2 = new float[6 * 7];
    float* bias1 = new float[7];

    float* in3 = new float[7 * 8];
    float* bias2 = new float[8];

    float* out = new float[5 * 8];
    float* out_ref = new float[5 * 8];

    srand48(time(NULL));
    for (size_t i = 0; i < 5 * 4; i++) {
        in0[i] = (float)drand48();
    }

    for (size_t i = 0; i < 4 * 6; i++) {
        in1[i] = (float)drand48();
    }

    for (size_t i = 0; i < 6 * 7; i++) {
        in2[i] = (float)drand48();
    }

    for (size_t i = 0; i < 7 * 8; i++) {
        in3[i] = (float)drand48();
    }

    for (size_t i = 0; i < 6; i++) {
        bias0[i] = (float)(i + 1);
    }

    for (size_t i = 0; i < 7; i++) {
        bias1[i] = (float)drand48();
    }

    for (size_t i = 0; i < 8; i++) {
        bias2[i] = (float)drand48();
    }

    for (size_t i = 0; i < 5 * 8; i++) {
        out[i] = 0.0f;
        out_ref[i] = 0.0f;
    }

    std::vector<void*> inputs = {static_cast<void*>(in0),
                                 static_cast<void*>(in1),
                                 static_cast<void*>(in2),
                                 static_cast<void*>(in3)};
    std::vector<void*> biases = {static_cast<void*>(bias2),
                                 static_cast<void*>(bias1),
                                 static_cast<void*>(bias0)};

    tree.execute(inputs, biases, out);

    // Reference calulations
    // first contraction in0, in1 -> out_int0
    float* out_int0 = new float[5 * 6];
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 6; j++) {
            out_int0[j * 5 + i] = bias0[j];
        }
    }
    gemm_ref(in0, in1, out_int0, 5, 6, 4, 5, 4, 5);

    // second contraction out_int0, in2 -> out_int1
    float* out_int1 = new float[5 * 7];
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 7; j++) {
            out_int1[j * 5 + i] = bias1[j];
        }
    }
    gemm_ref(out_int0, in2, out_int1, 5, 7, 6, 5, 6, 5);

    // final contraction out_int1, in3 -> out
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 8; j++) {
            out_ref[j * 5 + i] = bias2[j];
        }
    }
    gemm_ref(out_int1, in3, out_ref, 5, 8, 7, 5, 7, 5);

    double error = 0;
    for (size_t i = 0; i < 5 * 6; i++) {
        error += std::abs(out[i] - out_ref[i]);
    }
    std::cout << "Error: " << error << std::endl;
    REQUIRE(error < 1e-5);

    tree.delete_tree();
}

TEST_CASE("Einsum::Trees::EinsumTrees::simple binary operation with relu", "[Einsum][Trees][EinsumTrees]") {
    std::string str_repr = "[1,0],[2,1]->[2,0]r";
    EinsumTree tree = EinsumTree(str_repr, {7, 46, 88});
    tree.optimize();
    tree.lower();
    tree.print();
    float* in0 = new float[7 * 46];
    float* in1 = new float[46 * 88];
    float* out = new float[7 * 88];
    float* out_ref = new float[7 * 88];

    srand48(time(NULL));
    for (size_t i = 0; i < 7 * 46; i++) {
        in0[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < 46 * 88; i++) {
        in1[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < 7 * 88; i++) {
        out[i] = 0.0f;
        out_ref[i] = 0.0f;
    }
    tree.execute({in0, in1}, {}, out);

    gemm_ref(in0, in1, out_ref, 7, 88, 46, 7, 46, 7);

    for (int i = 0; i < 7 * 88; i++) {
        out_ref[i] = std::fmax(0.0f, out_ref[i]);
    }

    double error = 0;
    for (size_t i = 0; i < 7 * 88; i++) {
        error += std::abs(out[i] - out_ref[i]);
    }
    std::cout << "Error: " << error << std::endl;
    REQUIRE(error < 1e-5);

    tree.delete_tree();
}