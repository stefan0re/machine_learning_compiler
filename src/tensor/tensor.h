// tensor.h

#ifndef TENSOR_H
#define TENSOR_H

#include <sstream>
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

    std::vector<DimInfo> id;

    // constructor taking n dimension sizes
    // template definitions must be visible to every translation unit that uses them,
    // so define it in the header
    template <typename... Dims>
    Tensor(Dims... dims) {
        // store each dimension size in a vector
        std::vector<int> sizes = {dims...};

        setup(sizes);
    }

    Tensor(std::vector<u_int32_t> dims);

    void swap(int i, int j);
    void info() const;
    std::string info_str() const;

   private:
    void setup(std::vector<int> sizes);
};

#endif  // TENSOR_H
