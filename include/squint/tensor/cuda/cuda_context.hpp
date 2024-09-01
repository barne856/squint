#ifndef SQUINT_TENSOR_CUDA_CONTEXT_HPP
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace squint::cuda {

class cuda_context {
  public:
    static auto instance() -> cuda_context & {
        static cuda_context instance;
        return instance;
    }

    [[nodiscard]] auto cublas_handle() const -> cublasHandle_t { return cublas_handle_; }

    // Delete copy constructor and assignment operator
    cuda_context(const cuda_context &) = delete;
    auto operator=(const cuda_context &) -> cuda_context & = delete;

    // Delete move constructor and assignment operator
    cuda_context(cuda_context &&) = delete;
    auto operator=(cuda_context &&) -> cuda_context & = delete;

  private:
    cuda_context() {
        cublasStatus_t status = cublasCreate(&cublas_handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
    }

    ~cuda_context() { cublasDestroy(cublas_handle_); }

    cublasHandle_t cublas_handle_{};
};

} // namespace squint::cuda
#endif // SQUINT_TENSOR_CUDA_CONTEXT_HPP
