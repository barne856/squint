import numpy as np
import cupy as cp
import time

def time_function(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, (end - start) * 1000  # Convert to milliseconds

def main():
    # Create large matrices
    size = 10000
    a_cpu = np.arange(1, size*size + 1, dtype=np.float32).reshape(size, size)
    b_cpu = np.arange(1, size*size + 1, dtype=np.float32).reshape(size, size)

    # CPU multiplication
    _, cpu_time = time_function(np.dot, a_cpu, b_cpu)
    print(f"CPU multiplication time: {cpu_time:.2f} ms")

    # Transfer to GPU
    transfer_to_gpu_start = time.time()
    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    transfer_to_gpu_end = time.time()
    transfer_to_gpu_time = (transfer_to_gpu_end - transfer_to_gpu_start) * 1000
    print(f"Transfer to GPU time: {transfer_to_gpu_time:.2f} ms")

    # GPU multiplication
    _, gpu_time = time_function(cp.dot, a_gpu, b_gpu)
    print(f"GPU multiplication time: {gpu_time:.2f} ms")

    # Transfer from GPU
    transfer_from_gpu_start = time.time()
    _ = cp.asnumpy(a_gpu.dot(b_gpu))
    transfer_from_gpu_end = time.time()
    transfer_from_gpu_time = (transfer_from_gpu_end - transfer_from_gpu_start) * 1000
    print(f"Transfer from GPU time: {transfer_from_gpu_time:.2f} ms")

    # Calculate and print total GPU time
    total_gpu_time = transfer_to_gpu_time + gpu_time + transfer_from_gpu_time
    print(f"Total GPU time (including transfers): {total_gpu_time:.2f} ms")

if __name__ == "__main__":
    main()