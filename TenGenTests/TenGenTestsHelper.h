#ifndef TenGenTestsHelper_H
#define TenGenTestsHelper_H

#include <array>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "TenGen.h"

using Brgemm = TenGen::MiniJit::Generator::Brgemm;
using TensorOperation = TenGen::Einsum::Backend::TensorOperation;
using namespace TenGen::Types;

namespace TenGenTestsHelper {

    /**
     * @brief Generates machine code for an assembly instruction in form of a string.
     *
     * @param instruction assembly instruction in form of a string.
     *
     * @return machine code encoding.
     **/
    uint32_t as(const std::string& instruction) {
        // write the instruction to a temporary assembly file
        std::ofstream asmFile("temp.s");
        asmFile << ".text\n.global _start\n_start:\n    " << instruction << "\n";
        asmFile.close();

        // assemble it to an object file
        if (system("as temp.s -o temp.o") != 0) {
            throw std::runtime_error("Assembly failed");
        }

        // extract raw binary
        if (system("objcopy -O binary temp.o temp.bin") != 0) {
            throw std::runtime_error("Objcopy failed");
        }

        // read first 4 bytes of binary output
        std::ifstream binFile("temp.bin", std::ios::binary);
        std::array<char, 4> bytes{};
        binFile.read(bytes.data(), 4);
        binFile.close();

        // convert to uint32_t
        uint32_t result = static_cast<unsigned char>(bytes[0]) |
                          (static_cast<unsigned char>(bytes[1]) << 8) |
                          (static_cast<unsigned char>(bytes[2]) << 16) |
                          (static_cast<unsigned char>(bytes[3]) << 24);
        return result;
    }

    /**
     * @brief Generates binary representation of a decimal number.
     *
     * @param decimal decimal number.
     *
     * @return binary representation.
     **/
    std::bitset<32> get_binary(uint32_t decimal) {
        std::bitset<32> binary(decimal);
        return binary;
    }

    /**
     * @brief Checks if result and expected are equal, and prints and returns the result for unit testing.
     *
     * @param call respective assembly call.
     * @param result number to be tested.
     * @param expected correct encoding of the assembly call.
     *
     * @return if result==expected 0, else 1.
     **/
    int instr_is_correct(std::string call, uint32_t result, uint32_t expected) {
        bool match = (result == expected);
        std::cout << call << ": " << std::boolalpha << match << "\n"
                  << "result:  " << TenGenTestsHelper::get_binary(result) << "\n"
                  << "correct: " << TenGenTestsHelper::get_binary(expected)
                  << std::endl;
        return match ? 0 : -1;
    }

    /**
     @brief Generate a matrix with random numbers (or zeros) as elements.
     *
     * @param height m dimension of the produced matrix.
     * @param width n dimension of the produced matrix.
     * @param M pointer to the produced matrix.
     * @param set_zero if true, all elements of the produced matrix are set zero.
    */
    void generate_matrix(uint32_t height, uint32_t width, float* M, bool set_zero = false, bool visualization = false) {
        float MAX = 100;
        float MIN = -100;
        for (uint32_t i = 0; i < height; i++) {
            for (uint32_t j = 0; j < width; j++) {
                int index = j * height + i;
                float rand_float = static_cast<float>(rand()) / RAND_MAX;  // [0,1]
                rand_float = rand_float * (MAX - MIN) + MIN;
                if (visualization) {
                    rand_float = std::trunc(rand_float);
                }
                M[index] = (1 - (double)set_zero) * rand_float;
            }
        }
    }

    /**
     @brief Transpose (Identity) a given matrix.
    *
    * @param height m dimension of the matrix.
    * @param width n dimension of the matrix.
    * @param M pointer to the matrix that should transposed.
    * @param N pointer to the matrix that holds the final transposed matrix.
    */
    void transpose_matrix(uint32_t height, uint32_t width, float* M, float* N) {
        for (uint32_t i = 0; i < height; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                N[j * height + i] = M[i * width + j];
            }
        }
    }

    /**
     @brief Visualize matrix in terminal.
     *
     * @param height m dimension of the produced matrix.
     * @param width n dimension of the produced matrix.
     * @param M pointer to the produced matrix.
     * @param name contextual name of the matrix.
    */
    void visualize_matrix(uint32_t height, uint32_t width, float* M, std::string name) {
        std::cout << name << std::endl;
        for (uint32_t i = 0; i < height; i++) {
            for (uint32_t j = 0; j < width; j++) {
                int index = j * height + i;
                std::cout << M[index] << " ";
            }
            std::cout << std::endl;
        }
    }

    /**
     @brief Checks if result and expected are equal, and prints and returns the result for unit testing.
     *
     * @param height m dimension of the matrices.
     * @param width n dimension of the matrices.
     * @param M pointer to the first matrix.
     * @param C pointer to the second matrix.
     *
     * @return if worke correct 1, else 0.
    */
    bool compare_matrix(uint32_t height, uint32_t width, float* M, float* C) {
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

    void gemm_ref(float const* i_a,
                  float const* i_b,
                  float* io_c,
                  int64_t i_m,
                  int64_t i_n,
                  int64_t i_k,
                  int64_t i_lda,
                  int64_t i_ldb,
                  int64_t i_ldc) {
        for (int l_m = 0; l_m < i_m; l_m++) {
            for (int l_n = 0; l_n < i_n; l_n++) {
                for (int l_k = 0; l_k < i_k; l_k++) {
                    io_c[(l_n * i_ldc) + l_m] += i_a[(l_k * i_lda) + l_m] * i_b[(l_n * i_ldb) + l_k];
                }
            }
        }
    }

    void brgemm_ref(float const* i_a,
                    float const* i_b,
                    float* io_c,
                    int64_t i_m,
                    int64_t i_n,
                    int64_t i_k_gemm,
                    int64_t i_k_br,
                    int64_t i_lda,
                    int64_t i_ldb,
                    int64_t i_ldc,
                    int64_t i_br_stride_a,
                    int64_t i_br_stride_b) {
        for (int64_t l_k_br = 0; l_k_br < i_k_br; l_k_br++) {
            for (int64_t l_m = 0; l_m < i_m; l_m++) {
                for (int64_t l_n = 0; l_n < i_n; l_n++) {
                    for (int64_t l_k_gemm = 0; l_k_gemm < i_k_gemm; l_k_gemm++) {
                        io_c[(l_n * i_ldc) + l_m] += i_a[(l_k_br * i_br_stride_a) + (l_k_gemm * i_lda) + l_m] * i_b[(l_k_br * i_br_stride_b) + (l_n * i_k_gemm) + l_k_gemm];
                    }
                }
            }
        }
    }

    void run_1_example_with_scalar(float* i_ten_1,
                                   float* i_ten_2,
                                   float* o_ten) {
        for (size_t l_oM = 0; l_oM < 32; ++l_oM) {
            for (size_t l_oN = 0; l_oN < 32; ++l_oN) {
                for (size_t l_oK = 0; l_oK < 8; ++l_oK) {
                    for (size_t l_iM = 0; l_iM < 32; ++l_iM) {
                        for (size_t l_iN = 0; l_iN < 32; ++l_iN) {
                            for (size_t l_iK = 0; l_iK < 32; ++l_iK) {
                                size_t l_idx_1 = l_oM * 8192 + l_oK * 1024 + l_iK * 32 + l_iM;
                                size_t l_idx_2 = l_oN * 8192 + l_oK * 1024 + l_iN * 32 + l_iK;
                                size_t l_idx_out = l_oM * 32768 + l_oN * 1024 + l_iN * 32 + l_iM;

                                o_ten[l_idx_out] += i_ten_1[l_idx_1] * i_ten_2[l_idx_2];
                            }
                        }
                    }
                }
            }
        }
    }

    void run_1_example_with_gemm(float* i_ten_1,
                                 float* i_ten_2,
                                 float* o_ten) {
        Brgemm l_brgemm;
        l_brgemm.generate(32, 32, 32, 1, 0, 0, 0, dtype_t::fp32);
        Brgemm::kernel_t l_kernel = l_brgemm.get_kernel();

        for (size_t l_oM = 0; l_oM < 32; ++l_oM) {
            for (size_t l_oN = 0; l_oN < 32; ++l_oN) {
                for (size_t l_oK = 0; l_oK < 8; ++l_oK) {
                    float* l_a = &i_ten_1[l_oM * 8192 + l_oK * 1024];
                    float* l_b = &i_ten_2[l_oN * 8192 + l_oK * 1024];
                    float* l_c = &o_ten[l_oM * 32768 + l_oN * 1024];
                    l_kernel(l_a, l_b, l_c, 32, 32, 32, 8192, 8192);
                }
            }
        }
    }

    void run_example_with_einsum(float* i_ten_1,
                                 float* i_ten_2,
                                 float* o_ten,
                                 bool br_gemm,
                                 bool i_relu,
                                 bool i_first_touch_zero) {
        TensorOperation l_tensor_op;

        dtype_t l_dtype = dtype_t::fp32;
        prim_t l_prim_first_touch = i_first_touch_zero ? prim_t::zero : prim_t::none;
        prim_t l_prim_main = br_gemm ? prim_t::brgemm : prim_t::gemm;
        prim_t l_prim_last_touch = i_relu ? prim_t::relu : prim_t::none;

        std::vector<dim_t> l_dim_types = {dim_t::m,
                                          dim_t::n,
                                          dim_t::k,
                                          dim_t::m,
                                          dim_t::n,
                                          dim_t::k};

        std::vector<exec_t> l_exec_types = {exec_t::seq,
                                            exec_t::seq,
                                            br_gemm ? exec_t::prim : exec_t::seq,
                                            exec_t::prim,
                                            exec_t::prim,
                                            exec_t::prim};

        std::vector<int64_t> l_dim_sizes = {32, 32, 8, 32, 32, 32};

        std::vector<int64_t> l_strides_in0 = {8192, 0, 1024, 1, 0, 32};
        std::vector<int64_t> l_strides_in1 = {0, 8192, 1024, 0, 32, 1};
        std::vector<int64_t> l_strides_out = {32768, 1024, 0, 1, 32, 0};

        // Setup the tensor operation
        auto l_error = l_tensor_op.setup(l_dtype,
                                         l_prim_first_touch,
                                         l_prim_main,
                                         l_prim_last_touch,
                                         l_dim_types,
                                         l_exec_types,
                                         l_dim_sizes,
                                         l_strides_in0,
                                         l_strides_in1,
                                         l_strides_out);

        l_tensor_op.execute(i_ten_1, i_ten_2, o_ten);
    }

    bool check_diff(float* i_ten_1,
                    float* i_ten_2,
                    size_t i_size) {
        bool l_equal = true;
        double l_max_diff = 0.0;
        for (size_t i = 0; i < i_size; ++i) {
            if (std::abs(i_ten_1[i] - i_ten_2[i]) > 1e-3f) {
                l_equal = false;
                l_max_diff = std::max(l_max_diff, static_cast<double>(std::abs(i_ten_1[i] - i_ten_2[i])));
            }
        }
        if (l_equal) {
            return true;
        } else {
            return false;
        }
    }

}  // namespace TenGenTestsHelper

#endif  // TenGenTestsHelper_H