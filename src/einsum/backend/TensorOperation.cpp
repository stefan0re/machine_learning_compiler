#include "TensorOperation.h"

/**
 * General-purpose loop implementation featuring first and last touch operations.
 * No threading is applied.
 *
 * @param id_loop      Dimension id of the loop which is executed.
 * @param ptr_in0      Pointer to the first input tensor's data.
 * @param ptr_in1      Pointer to the second input tensor's data (use nullptr if unary).
 * @param ptr_out      Pointer to the output tensor's data.
 * @param first_access True if first time accessing data of output tensor.
 * @param last_access  True if last time accessing data of output tensor.
 **/
void execute_iter(int64_t id_loop,
                  char const* ptr_in0,
                  char const* ptr_in1,
                  char* ptr_out,
                  bool first_access,
                  bool last_access) {
    int64_t l_size = m_loop_sizes[id_loop];

    for (int64_t l_it = 0; l_it < l_size; l_it++) {
        // derive if this is first or last access to the output block

        // update pointer with strides

        if (id_loop + 1 < m_id_first_primitive_loop) {
            // recursive function call
        } else {
            // call first touch kernel if necessary

            // call main kernel

            // call last touch kernel if necessary
        }
    }
}