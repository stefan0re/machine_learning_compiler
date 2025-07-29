#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include "../../src/tensor/tensor.h"

// ReLU activation
void relu(Tensor& vec) {
    for (size_t i = 0; i < vec.size; ++i) {
        vec.data[i] = std::max(0.0f, vec.data[i]);
    }
}

// Matrix-vector multiplication: y = x * W^T + b
// in row-major access pattern
void matmul_add(const Tensor& x, const Tensor& W, const Tensor& b, Tensor& out) {
    /*
    W = [w1.1, w1.2, w1.3, w1.4,        [x1,   [y1,
         w2.1, w2.2, w2.3, w2.4,   *     x2, =  y2,
         w3.1, w3.2, w3.3, w3.4,         x3,    y3,
                ......          ]        x4]    ...]

    -> y1 = w1.1 * x1 + w1.2 + w1.3 * w3 + w1.4 * w4 + b1.1
    -> Tensor(4, 64) -> stride: (64, 1)
    */
    for (size_t i = 0; i < out.size; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < x.size; ++j) {
            // for each value in X and each row in W
            sum += x.data[j] * W.data[64 * i + j];
        }
        out.data[i] = sum + b.data[i];
    }
}

// Forward pass of BasicNet
void forward(const Tensor& input,                 // size = b = 4
             const Tensor& W1, const Tensor& b1,  // W1: [c][b], b1: [c]
             const Tensor& W2, const Tensor& b2,  // W2: [d][c], b2: [d]
             const Tensor& W3, const Tensor& b3,  // W3: [e][d], b3: [e]
             Tensor& output)                      // output: size = e
{
    // Temporary tensors
    Tensor z1 = Tensor(64);
    Tensor z2 = Tensor(16);

    // fc1
    matmul_add(input, W1, b1, z1);
    relu(z1);

    // fc2
    matmul_add(z1, W2, b2, z2);
    relu(z2);

    // fc3
    matmul_add(z2, W3, b3, output);

    delete[] z1.data;
    delete[] z2.data;
}

TEST_CASE("Model::BasicNet::Output", "[Model][BasicNet][Output]") {
    // load the trained model
    std::vector<Tensor> model = Tensor::from_torchpp("../../python/data/model.torchpp", 4);

    Tensor W1 = model[0];
    Tensor b1 = model[1];
    Tensor W2 = model[2];
    Tensor b2 = model[3];
    Tensor W3 = model[4];
    Tensor b3 = model[5];

    // load example input and output
    Tensor example = Tensor::from_csv("../../python/data/example.csv");

    // input
    Tensor input = Tensor(4);
    input.data = new float[4]{example.data[0], example.data[1], example.data[2], example.data[3]};

    // output
    Tensor output = Tensor(3);
    Tensor output_ref = Tensor(3);
    output_ref.data = new float[3]{example.data[4], example.data[5], example.data[6]};

    // DEBUG
    input.print();
    output_ref.print();
    b1.print();

    forward(input, W1, b1, W2, b2, W3, b3, output);

    REQUIRE(output.compare(output_ref));
}
