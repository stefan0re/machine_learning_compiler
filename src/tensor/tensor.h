// tensor.h

#ifndef TENSOR_H
#define TENSOR_H

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

    template <typename... Dims>
    Tensor(Dims... dims);

    void swap(int i, int j);
    void info() const;
};

#endif  // TENSOR_H
