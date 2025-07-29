// tensor.h

#ifndef TENSOR_H
#define TENSOR_H

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

class Tensor {
   public:
    struct DimInfo {
        int dim_t = 99;
        int dim_sizes;
        int stride;
        int loop_id;
        int exec_t = 99;
    };

    size_t size;
    float* data = nullptr;

    std::vector<DimInfo> id;

    // constructor taking n dimension sizes
    // template definitions must be visible to every translation unit that uses them,
    // so define it in the header
    template <typename... Dims,
              typename = std::enable_if_t<(std::conjunction_v<std::is_integral<Dims>...>)>>
    Tensor(Dims... dims) {
        // store each dimension size in a vector
        std::vector<int> sizes = {dims...};

        setup(sizes);

        size = (static_cast<size_t>(dims) * ...);
        data = new float[size];
    }

    Tensor(std::vector<u_int32_t> dims);

    void swap(int i, int j);
    void info() const;
    std::string info_str() const;

    /**
     * @brief Loads a sequence of Tensors from a .torchpp file.
     *
     * This function parses a specially formatted file.It assumes
     * alternating lines of weights (matrices) and biases (vectors).
     *
     * - Even-numbered lines (0-based) are interpreted as weight matrices. The number of rows
     *   is given by `in_size`, and columns are inferred from the total number of elements.
     * - Odd-numbered lines are interpreted as bias vectors.
     *
     * Each parsed line is converted into a Tensor, with appropriate allocation and copying
     * of the float values. The function returns a vector of these Tensors, representing
     * the layers of a neural network (e.g., [weight, bias, weight, bias, ...]).
     *
     * @param path The file path to the torchpp file containing the weights and biases.
     * @param in_size The initial input size (number of features) to infer weight matrix dimensions.
     * @return std::vector<Tensor> A vector containing the parsed weight and bias tensors.
     */
    static std::vector<Tensor> from_torchpp(std::string path, int in_size);

    /**
     * @brief Creates a Tensor by loading float values from a CSV file.
     *
     * This function reads a CSV file specified by the given path. It parses each line
     * assuming comma-separated values, converts valid tokens to floats, and ignores
     * non-numeric entries. The tensor is constructed with dimensions inferred from
     * the number of lines and the number of elements per line.
     *
     * @param path The path to the CSV file.
     * @return Tensor A tensor populated with the parsed float values.
     */
    static Tensor from_csv(std::string path);

    /**
     * @brief Compare this tensor with anothe
     *
     * @param tensor The tensor that should be compared with.
     * @param delta A small margin to compensate floating point calculation errors.
     * @return bool True if the tensors are equal.
     */
    bool compare(Tensor& tensor, float delta);

    /**
     * @brief Print the raw data of the Tensor
     */
    void print();

   private:
    void setup(std::vector<int> sizes);
};

#endif  // TENSOR_H
