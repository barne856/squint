#ifndef SQUINT_TENSOR_CUDA_CONTEXT_HPP
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace squint::cuda {

class CudaContext {
  public:
    static auto instance() -> CudaContext & {
        static CudaContext instance;
        return instance;
    }

    [[nodiscard]] auto cublas_handle() const -> cublasHandle_t { return cublas_handle_; }

    // Delete copy constructor and assignment operator
    CudaContext(const CudaContext &) = delete;
    auto operator=(const CudaContext &) -> CudaContext & = delete;

    // Delete move constructor and assignment operator
    CudaContext(CudaContext &&) = delete;
    auto operator=(CudaContext &&) -> CudaContext & = delete;

  private:
    CudaContext() {
        cublasStatus_t status = cublasCreate(&cublas_handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
    }

    ~CudaContext() { cublasDestroy(cublas_handle_); }

    cublasHandle_t cublas_handle_{};
};

} // namespace squint::cuda
#endif // SQUINT_TENSOR_CUDA_CONTEXT_HPP
