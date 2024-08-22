
Performance Considerations
==========================


To get the best performance out of SQUINT:

1. Use fixed-size tensors when dimensions are known at compile-time. This allows for more aggressive compiler optimizations.

2. Choose the appropriate BLAS backend for your hardware and use case:
   - Use Intel MKL on Intel processors for best performance
   - Use OpenBLAS for good performance on a variety of architectures
   - Use the NONE backend only when portability is the highest priority

3. Prefer views over copies for subsections of tensors to avoid unnecessary memory allocations and copies.

4. Disable error checking in performance-critical code paths once you're confident in your implementation's correctness.

