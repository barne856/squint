#ifndef SQUINT_TENSOR_CUDA_SCALAR_HPP

template <typename T>
void scalar_multiplication(T scalar, T *output, const T *a, const unsigned long *dims, const unsigned long *strides_out,
                           const unsigned long *strides_a, unsigned long num_dims, unsigned long total_size);

#endif // SQUINT_TENSOR_CUDA_SCALAR_HPP
