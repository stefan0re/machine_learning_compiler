#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>

#include "../src/mini_jit/"

void reference_mat_mul(float const* a,
                       float const* b,
                       float* c,
                       int64_t n,
                       int64_t m,
                       int64_t k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                c[(j * m) + i] += a[(l * m) + i] * b[(j * k) + l];
            }
        }
    }
}

void visualize_matix(float const* c,
                     int64_t height,
                     int64_t width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = j * height + i;
            std::cout << c[index] << " ";
        }
        std::cout << std::endl;
    }
}

void get_matrices(float* a,
                  float* b,
                  float* c,
                  float* c_ref,
                  int64_t n,
                  int64_t m,
                  int64_t k,
                  bool visualization = false) {
    float MAX = 100.f;
    // fill a
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            int a_index = j * m + i;
            a[a_index] = (1 - (double)visualization) * static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / MAX)) + (double)visualization * i;
        }
    }

    // fill b
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            int b_index = j * k + i;
            b[b_index] = (1 - (double)visualization) * static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / MAX)) + (double)visualization * j;
        }
    }

    // fill c
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int c_index = j * m + i;
            float element = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / MAX));
            c[c_index] = (1 - (double)visualization) * element;
            c_ref[c_index] = (1 - (double)visualization) * element;
        }
    }
}

int benchmark(void (*jit_func)(float const*, float const*, float*, int64_t, int64_t, int64_t));

int main() {
    srand(static_cast<unsigned>(time(0)));

    return 0;
}
