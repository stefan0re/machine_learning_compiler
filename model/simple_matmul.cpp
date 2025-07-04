#include <chrono>
#include <cstdlib>
#include <iostream>

#include "../src/einsum/trees/einsum_trees.h"
#include "../src/mini_jit/include/gemm_ref.h"
#include "../src/tensor/tensor.h"

/** GEMM ab,ca-> cb,
 * Sizes: 4,2,3
 */
using namespace einsum::trees;

int main() {
    std::string tree_string = "[1,0],[2,1]->[2,0]r]";

    EinsumTree tree = EinsumTree(tree_string, {2, 4, 3}, false);

    tree.optimize();
    tree.lower();
    tree.print();

    float* in0 = new float[2 * 4];
    float* in1 = new float[3 * 4];
    float* out = new float[2 * 3];
    float* out_ref = new float[2 * 3];
    float* bias = nullptr;

    srand(time(NULL));

    for (size_t i = 0; i < 2 * 4; i++) {
        in0[i] = (float)drand48() * 10 - 5;
    }
    for (size_t i = 0; i < 3 * 4; i++) {
        in1[i] = (float)drand48() * 10 - 5;
    }

    for (size_t i = 0; i < 2 * 3; i++) {
        out[i] = 0.0;
        out_ref[i] = 0.0;
    }

    tree.execute({in0, in1}, {bias}, out);

    gemm_ref(in0, in1, out_ref, 2, 3, 4, 2, 4, 2);

    double error = 0.0;
    size_t count_error = 0;
    for (size_t i = 0; i < 2 * 3; i++) {
        error += fabs(out[i] - out_ref[i]);
        std::cout << "Value at index " << i << ": " << out[i] << " != " << out_ref[i] << std::endl;
        if (fabs(out[i] - out_ref[i]) > 1e-5) {
            count_error++;
        }
    }
    std::cout << "Error: " << error << std::endl;
}