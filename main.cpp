#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <squint/squint.hpp>

using namespace squint;

// cublasStatus_t cublasSgemm(cublasHandle_t handle,
//                            cublasOperation_t transa, cublasOperation_t transb,
//                            int m, int n, int k,
//                            const float           *alpha,
//                            const float           *A, int lda,
//                            const float           *B, int ldb,
//                            const float           *beta,
//                            float           *C, int ldc)

// cusolverStatus_t cusolverDnSSgels(
//         cusolverDnHandle_t      handle,
//         int                     m,
//         int                     n,
//         int                     nrhs,
//         float               *   dA,
//         int                     ldda,
//         float               *   dB,
//         int                     lddb,
//         float               *   dX,
//         int                     lddx,
//         void                *   dWorkspace,
//         size_t                  lwork_bytes,
//         int                 *   niter,
//         int                 *   dinfo);

// Error checking macro
#define cudaCheckError(ans)                                                                                            \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

auto main() -> int {
    // get cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // get cublas version
    int version;
    cublasGetVersion(handle, &version);
    std::cout << "CUBLAS version: " << version << std::endl;

    auto A = ndarr<4, 4>::arange(1, 1);
    auto B = ndarr<4, 4>::arange(1, 1);
    auto C = ndarr<4, 4>::zeros();

    // perform matrix multiplication using cublas
    auto A_device = A.to_device();
    auto B_device = B.to_device();
    auto C_device = C.to_device();

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, &alpha, A_device.data(), 4, B_device.data(), 4, &beta,
                C_device.data(), 4);

    // copy result back to host
    auto C_host = C_device.to_host();

    std::cout << C_host << std::endl;

    auto A2 = ndarr<2, 2>{3, 1, 1, 2};
    auto B2 = ndarr<2>{9, 8};
    auto C2 = ndarr<2>::zeros();
    auto A2_device = A2.to_device();
    auto B2_device = B2.to_device();
    auto C2_device = C2.to_device();

    // get cusolver handle
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);

    // solve the system of linear equations using cusolver sgels
    int m = 2;    // number of rows of A
    int n = 2;    // number of columns of A
    int nrhs = 1; // number of right hand sides
    int ldda = m; // leading dimension of A
    int lddb = m; // leading dimension of B
    int lddx = n; // leading dimension of X (output)
    int niter = 0;
    int dinfo = 0;

    // Step 1: Query the required workspace size
    size_t lwork_bytes = 0;
    cusolverStatus_t status =
        cusolverDnSSgels_bufferSize(cusolver_handle, m, n, nrhs, A2_device.data(), ldda, B2_device.data(), lddb,
                                    C2_device.data(), lddx, nullptr, &lwork_bytes);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cusolverDnSSgels_bufferSize failed with error code " << status << std::endl;
        return -1;
    }

    // print workspace size
    std::cout << "Workspace size: " << lwork_bytes << " bytes" << std::endl;

    // Allocate device workspace
    float *dWorkspace = nullptr;
    cudaCheckError(cudaMalloc(&dWorkspace, lwork_bytes));

    // Step 2: Solve the system
    status = cusolverDnSSgels(cusolver_handle, m, n, nrhs, A2_device.data(), ldda, B2_device.data(), lddb,
                              C2_device.data(), lddx, dWorkspace, lwork_bytes, &niter, &dinfo);

    // Check for errors
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cusolverDnSSgels failed with error code " << status << std::endl;
        cudaFree(dWorkspace);
        cusolverDnDestroy(cusolver_handle);
        return -1;
    }

    // Check dinfo
    if (dinfo != 0) {
        std::cerr << "The algorithm failed to compute a solution. dinfo = " << dinfo << std::endl;
    } else {
        std::cout << "Solution found in " << niter << " iterations." << std::endl;
    }

    // Copy result back to host
    auto C2_host = ndarr<2>::zeros(); // Ensure C2_host is properly allocated
    cudaCheckError(cudaMemcpy(C2_host.data(), C2_device.data(), sizeof(float) * n * nrhs, cudaMemcpyDeviceToHost));

    // Print the result
    std::cout << "Solution:" << std::endl;
    std::cout << C2_host << std::endl;

    // Clean up
    cudaCheckError(cudaFree(dWorkspace));
    cusolverDnDestroy(cusolver_handle);

    return 0;
}