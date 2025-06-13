#ifndef TENGEN_TENSOR_H
#define TENGEN_TENSOR_H

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>

// e.g. Tensor<M, N, K> t = Tensor<M, N, K>(3, 4, 5)
template <size_t... d_t>
class Tensor {
   public:
    //
    // --------------------- Define some tensor attributes ---------------------
    //

    // store the rank of the tensor, e.g. Tensor<M, N, K>::rank  // == 3
    static constexpr size_t rank = sizeof...(d_t);

    // store the dimension types of the tensor
    static constexpr std::array<einsum::backend::TensorOperation::dim_t, rank> _dim_types = {d_t...};

    // sizes of each dimension (calculated in the constructor)
    std::vector<int64_t> _dim_sizes;

    // the data of the tensor (calculated in the constructor)
    std::vector<float> _data;

    // count of elements
    int64_t size;

    // calculated in the constructor
    std::vector<int64_t> strides;

    //
    // ---------------------------- Constructor ---------------------------------
    //

    template <typename... sizes>
    Tensor(sizes... sizes, bool fill_random = false) {
        // get the sizes of the tensor
        size = (static_cast<int64_t>(sizes) * ...);

        _dim_sizes.resize(sizeof...(sizes));
        _dim_sizes = {static_cast<int>(sizes)...};

        // set the amount of elements in the tensor
        _data.resize(size);

        // store the stride of each dimension
        // e.g. Tensor<float, 2, 3, 4>::strides == {12, 4, 1}
        // S2 = 1                   last dimension stride is always 1
        // S1 = D2 * S2 = 4 * 1 = 4
        // S0 = D1 * S1 = 3 * 4 = 12
        int64_t stride = 1;
        for (size_t i = rank; i-- > 0;) {
            strides[i].push_back(stride);
            stride *= _dim_sizes[i];
        }

        // if fill_random is true, fill the tensor with random values
        if (fill_random) {
            fillRandom();
        }
    }

    //
    // ------------------------------ Overloads ---------------------------------
    //

    // overload of the subscript operator and
    // returns a reference to the underlying data element at that index (flat)
    float& operator[](size_t index) {
        return _data[index];
    }

    bool operator==(const Tensor& other) const {
        // Compare data
        return _data == other._data;
    }

    //
    // ------------------------------ Functions ---------------------------------
    //

    // multidimensional access using indices
    template <typename... Indices>
    auto at(Indices... indices) {
        // store the indices in an array
        std::array<size_t, rank> idx = {static_cast<size_t>(indices)...};
        size_t flat_index = 0;

        // for each index, calculate the flat index
        // e.g. index = i * S0 + j * S1 + k * S2 + ... + n * Sn
        for (size_t i = 0; i < rank; ++i) {
            flat_index += idx[i] * strides[i];
        }

        return _data[flat_index];
    }

    // fills the tensor with random values (integers or floats)
    void fillRandom() {
        for (size_t i = 0; i < size; ++i) {
            _data[i] = srand() / RAND_MAX;  // e.g. 0.0 - 1.0
        }
    }

    //
    // --------------------------- Export / Import ------------------------------
    //

    // exports the pointer to the underlying data
    char* getPointer() {
        return reinterpret_cast<char*>(_data.data());
    }

    // create a real copy from the fastor tensor
    void importData(const std::vector<float>& new_data) {
        _data = new_data;
    }

    //
    // -------------------------------- DEBUG -----------------------------------
    //

    // DEBUG function to print the infos of the tensor
    void info() const {
        std::cout << "Tensor Info:" << std::endl;
        std::cout << "  Rank: " << rank << std::endl;
        std::cout << "  Dimension Types: ";
        for (const auto& dim : _dim_types) {
            std::cout << dim << " ";
        }
        std::cout << "  Shape: ";
        for (const auto& dim : _dim_sizes) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "  Strides: ";
        for (const auto& stride : strides) {
            std::cout << stride << " ";
        }
        std::cout << std::endl;
        std::cout << "  Size: " << size << std::endl;
    }
};

#endif  // TENSOR_TENSOR_TENSOR_H