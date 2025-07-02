#include "test_utils.h"
void test::matmul::generate_matrix(uint32_t height, uint32_t width, float* M, bool set_zero, bool visualization) {
    float MAX = 100;
    float MIN = -100;
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int index = j * height + i;
            float rand_float = static_cast<float>(rand()) / RAND_MAX;
            rand_float = rand_float * (MAX - MIN) + MIN;
            if (visualization) {
                rand_float = std::trunc(rand_float);
            }

            if (set_zero) {
                M[index] = 0.0f;
            } else {
                M[index] = rand_float;
            }
        }
    }
}

bool test::matmul::compare_matrix(uint32_t height, uint32_t width, float* M, float* C) {
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int index = j * height + i;
            float epsilon = 1e-5;
            if (std::fabs(M[i] - C[i]) > epsilon) {
                std::cout << "Matrices are not equal in element: " << index << " M: " << M[index] << ", C: " << C[index] << std::endl;
                return false;
            }
        }
    }
    return true;
}

void test::matmul::print_matrix(uint32_t height, uint32_t width, float* M) {
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int index = j * height + i;
            std::cout << M[index] << " ";
        }
        std::cout << std::endl;
    }
}