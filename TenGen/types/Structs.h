#ifndef TENGEN_STRUCTS_H
#define TENGEN_STRUCTS_H

#include <cstdint>
#include <span>
#include <vector>

#include "TenGen/types/Types.h"

using namespace TenGen::Types;

namespace TenGen::Structs {
    struct KernelSize {
        int M;
        int N;
    };

    struct KernelSizes {
        KernelSize kernel1;
        KernelSize kernel2;
        KernelSize kernel3;
        KernelSize kernel4;
    };

    struct AreaDefinition {
        int32_t m;
        int32_t n;
        int32_t m_iters;
        int32_t n_iters;
        uint32_t offset;
        KernelSize kernelsize;
    };

    struct TensorConfig {
        // scalars
        dtype_t dtype;
        prim_t prim_first_touch;
        prim_t prim_main;
        prim_t prim_last_touch;
        int64_t id_first_primitive_loop;

        int64_t id_prim_m;
        int64_t id_prim_n;
        int64_t id_prim_k = 0;
        int64_t id_prim_br_size = -1;

        int64_t lda;
        int64_t ldb;
        int64_t ldc;
        int64_t in0_br_stride;
        int64_t in1_br_stride;

        // owned storage
        std::vector<dim_t> _dim_types_storage;
        std::vector<exec_t> _exec_types_storage;
        std::vector<int64_t> _dim_sizes_storage;
        std::vector<int64_t> _strides_in0_storage;
        std::vector<int64_t> _strides_in1_storage;
        std::vector<int64_t> _strides_out_storage;
        std::vector<int64_t> _loop_sizes_storage;

        // views (spans)
        std::span<const dim_t> dim_types;
        std::span<const exec_t> exec_types;
        std::span<const int64_t> dim_sizes;
        std::span<const int64_t> strides_in0;
        std::span<const int64_t> strides_in1;
        std::span<const int64_t> strides_out;
        std::span<const int64_t> loop_sizes;
    };

}  // namespace TenGen::Structs

#endif  // TENGEN_STRUCTS_H
