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

TEST_CASE("Einsum::Trees::EinsumTrees::Large Tree Example 1 Lower", "[Einsum][Trees][EinsumTrees]") {
    std::string str_repr = "[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]";
    EinsumTree tree = EinsumTree(str_repr, {100, 72, 128, 128, 3, 71, 305, 32, 3});
    tree.lower();
    tree.print();

    float* in0 = new float[3 * 3];
    float* in1 = new float[32 * 128 * 3];
    float* in2 = new float[128 * 305 * 32];
    float* in3 = new float[72 * 71 * 305];
    float* in4 = new float[100 * 71];
    float* out = new float[100 * 72 * 128 * 128 * 3];
    float* out_ref = new float[100 * 72 * 128 * 128 * 3];

    srand48(time(NULL));
    for (size_t i = 0; i < 3 * 3; i++) {
        in0[i] = (float)drand48();
    }

    for (size_t i = 0; i < 32 * 128 * 3; i++) {
        in1[i] = (float)drand48();
    }

    for (size_t i = 0; i < 128 * 305 * 32; i++) {
        in2[i] = (float)drand48();
    }

    for (size_t i = 0; i < 72 * 71 * 305; i++) {
        in3[i] = (float)drand48();
    }

    for (size_t i = 0; i < 100 * 71; i++) {
        in4[i] = (float)drand48();
    }

    for (size_t i = 0; i < 100 * 72 * 128 * 128 * 3; i++) {
        out[i] = 0.0f;
        out_ref[i] = 0.0f;
    }

    tree.execute({in0, in1, in2, in3, in4}, {}, out);

    // Reference calculations
    // first contraction in0, in1 -> out_int0
    float* out_int0 = new float[32 * 128 * 4];

    for (size_t n_0 = 0; n_0 < 32; n_0++) {
        for (size_t m = 0; m < 3; m++) {
            for (size_t n_1 = 0; n_1 < 128; n_1++) {
                for (size_t k = 0; k < 3; k++) {
                    size_t idx_in0 = n_0 * 0 + m * 1 + n_1 * 0 + k * 3;
                    size_t idx_in1 = n_0 * 384 + m * 0 + n_1 * 3 + k * 1;
                    size_t idx_out = n_0 * 384 + m * 1 + n_1 * 3 + k * 0;

                    out_int0[idx_out] += in0[idx_in0] * in1[idx_in1];
                }
            }
        }
    }

    // second contraction in2, in3 -> out_int1
    float* out_int1 = new float[72 * 128 * 71 * 32];

    for (size_t m_0 = 0; m_0 < 128; m_0++) {
        for (size_t n_0 = 0; n_0 < 72; n_0++) {
            for (size_t m_1 = 0; m_1 < 32; m_1++) {
                for (size_t n_1 = 0; n_1 < 71; n_1++) {
                    for (size_t k = 0; k < 305; k++) {
                        size_t idx_in2 = m_0 * 9760 + n_0 * 0 + m_1 * 1 + n_1 * 0 + k * 32;
                        size_t idx_in3 = m_0 * 0 + n_0 * 21655 + m_1 * 0 + n_1 * 305 + k * 1;
                        size_t idx_out = n_0 * 128 * 71 * 32 + m_0 * 71 * 32 + n_1 * 32 + m_1 * 1 + k * 0;

                        out_int1[idx_out] += in2[idx_in2] * in3[idx_in3];
                    }
                }
            }
        }
    }

    // third contraction out_int1, in4 -> out
    float* out_int2 = new float[100 * 72 * 128 * 32];

    for (size_t m_0 = 0; m_0 < 72; m_0++) {
        for (size_t m_1 = 0; m_1 < 128; m_1++) {
            for (size_t m_2 = 0; m_2 < 32; m_2++) {
                for (size_t n = 0; n < 100; n++) {
                    for (size_t k = 0; k < 71; k++) {
                        size_t idx_int1 = m_0 * 128 * 32 * 71 + m_1 * 32 * 71 + m_2 * 1 + n * 0 + k * 32;
                        size_t idx_in4 = m_0 * 0 + m_1 * 0 + m_2 * 0 + n * 71 + k * 1;
                        size_t idx_out = m_0 * 32 * 128 + m_1 * 32 + m_2 * 1 + n * 72 * 32 * 128 + k * 0;

                        out_int2[idx_out] += out_int1[idx_int1] * in4[idx_in4];
                    }
                }
            }
        }
    }

    // final contraction out_int0, out_int2 -> out

    for (size_t n_0 = 0; n_0 < 100; n_0++) {
        for (size_t m_0 = 0; m_0 < 128; m_0++) {
            for (size_t n_1 = 0; n_1 < 72; n_1++) {
                for (size_t m_1 = 0; m_1 < 3; m_1++) {
                    for (size_t n_2 = 0; n_2 < 128; n_2++) {
                        for (size_t k = 0; k < 32; k++) {
                            size_t idx_int0 = n_0 * 0 + m_0 * 3 + n_1 * 0 + m_1 * 1 + n_2 * 0 + k * 128 * 3;
                            size_t idx_int2 = n_0 * 72 * 128 * 32 + m_0 * 0 + n_1 * 128 * 32 + m_1 * 0 + n_2 * 32 + k * 1;
                            size_t idx_out = n_0 * 72 * 128 * 128 * 3 + m_0 * 3 + n_1 * 128 * 128 * 3 + m_1 * 1 + n_2 * 128 * 3 + k * 0;

                            out_ref[idx_out] += out_int0[idx_int0] * out_int2[idx_int2];
                        }
                    }
                }
            }
        }
    }

    double error = 0;
    for (size_t i = 0; i < 100 * 72 * 128 * 128 * 3; i++) {
        error += std::abs(out[i] - out_ref[i]);
    }
    std::cout << "Error: " << error << std::endl;
    REQUIRE(error < 1e-5);

    delete[] in0;
    delete[] in1;
    delete[] in2;
    delete[] in3;
    delete[] in4;
    delete[] out;
    delete[] out_ref;
    delete[] out_int0;
    delete[] out_int1;
    delete[] out_int2;
    tree.delete_tree();
}

TEST_CASE("Einsum::Trees::EinsumTrees::simple permutation example", "[Einsum][Trees][EinsumTrees]") {
    std::cout << "Running first pbtc example..." << std::endl;
    std::string str_repr = "[0,1,2]->[2,0,1]";

    EinsumTree tree = EinsumTree(str_repr, {4, 3, 2});
    tree.lower();

    float* in = new float[4 * 3 * 2];
    float* out = new float[4 * 3 * 2];
    float out_ref[24] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,
                         2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};

    for (size_t i = 0; i < 24; i++) {
        in[i] = i + 1;
        out[i] = 0;
    }

    tree.execute({in}, {}, out);

    for (size_t i = 0; i < 24; i++) {
        REQUIRE(out[i] == out_ref[i]);
    }
}

