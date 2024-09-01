#include <cstdint>
#ifndef SQUINT_TENSOR_CUDA_ELEMENT_WISE_HPP

template <typename T>
void element_wise_addition(T *output, const T *a, const T *b, const unsigned long *dims,
                           const unsigned long *strides_out, const unsigned long *strides_a,
                           const unsigned long *strides_b, unsigned long num_dims, unsigned long total_size);

template <typename T>
void element_wise_subtraction(T *output, const T *a, const T *b, const unsigned long *dims,
                              const unsigned long *strides_out, const unsigned long *strides_a,
                              const unsigned long *strides_b, unsigned long num_dims, unsigned long total_size);

template <typename T>
void element_wise_equality(uint8_t *output, const T *a, const T *b, const unsigned long *dims,
                           const unsigned long *strides_out, const unsigned long *strides_a,
                           const unsigned long *strides_b, unsigned long num_dims, unsigned long total_size);

template <typename T>
void element_wise_inequality(uint8_t *output, const T *a, const T *b, const unsigned long *dims,
                             const unsigned long *strides_out, const unsigned long *strides_a,
                             const unsigned long *strides_b, unsigned long num_dims, unsigned long total_size);

template <typename T>
void element_wise_negation(T *output, const T *a, const unsigned long *dims, const unsigned long *strides_out,
                           const unsigned long *strides_a, unsigned long num_dims, unsigned long total_size);

#endif // SQUINT_TENSOR_CUDA_ELEMENT_WISE_HPP
