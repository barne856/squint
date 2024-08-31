#ifndef SQUINT_TENSOR_CUDA_ELEMENT_WISE_HPP
// NOLINTBEGIN

template <typename T>
void element_wise_addition(T *output, const T *a, const T *b, const unsigned long *dims,
                           const unsigned long *strides_out, const unsigned long *strides_a,
                           const unsigned long *strides_b, unsigned long num_dims, unsigned long total_size);

// NOLINTEND
#endif // SQUINT_TENSOR_CUDA_ELEMENT_WISE_HPP