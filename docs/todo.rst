TODO
====

- [ ] Optimize small vectors using SIMD intrinsics. E.g. vec4 should have data storage ``__m128`` or ``__m256d`` for float and double respectively.
- [ ] Optimize tensor math for SIMD intrinsics. E.g. matrix multiplication should use SIMD intrinsics for 4x4 matrix multiplication, cross product, etc.
- [ ] Optimize element access for 1D and 2D tensors.
- [ ] Basic tensor expression templates for fused and chained operations, element-wise, scalar, and matrix operations
