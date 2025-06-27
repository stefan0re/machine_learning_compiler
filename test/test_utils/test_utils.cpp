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
            M[index] = (1 - (double)set_zero) * rand_float;
        }
    }
}

bool test::matmul::compare_matrix(uint32_t height, uint32_t width, float* M, float* C) {
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            int index = j * height + i;
            if (M[index] != C[index]) {
                std::cout << "Matrices are not equal in element: " << index << " M: " << M[index] << ", C: " << C[index] << std::endl;
                return false;
            }
        }
    }
    return true;
}