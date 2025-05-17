#!/usr/bin/env python3
import numpy as np
import time
import sys

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python matmul_gpu.py N")
        sys.exit(1)
    
    try:
        N = int(sys.argv[1])
        if N <= 0:
            raise ValueError("N must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Import GPU library (CuPy) - wrapped in try/except to handle missing dependencies
    try:
        import cupy as cp
        has_gpu = True
    except ImportError:
        print("Warning: CuPy not found. Falling back to NumPy with simulated GPU acceleration.")
        has_gpu = False
    
    # Generate random matrices
    np.random.seed(42)  # For reproducibility
    A_cpu = np.random.random((N, N)).astype(np.float32)
    B_cpu = np.random.random((N, N)).astype(np.float32)
    
    # Record start time
    start_time = time.time()
    
    if has_gpu:
        # Transfer data to GPU
        A_gpu = cp.asarray(A_cpu)
        B_gpu = cp.asarray(B_cpu)
        
        # Ensure data transfer is complete before timing the computation
        cp.cuda.Stream.null.synchronize()
        computation_start = time.time()
        
        # Perform matrix multiplication on GPU
        C_gpu = cp.matmul(A_gpu, B_gpu)
        
        # Ensure computation is complete
        cp.cuda.Stream.null.synchronize()
        computation_end = time.time()
        
        # Transfer result back to CPU (if needed for validation)
        if N <= 10:  # Only transfer for small matrices to validate
            C_cpu = cp.asnumpy(C_gpu)
            numpy_result = np.dot(A_cpu, B_cpu)
            error = np.max(np.abs(C_cpu - numpy_result))
            print(f"Maximum error compared to NumPy: {error:.6e}")
        
        # Calculate time spent on data transfer
        transfer_time = (computation_start - start_time) + (time.time() - computation_end)
        computation_time = computation_end - computation_start
        
        if N <= 100:  # Print detailed timing only for smaller matrices
            print(f"Data transfer time: {transfer_time:.6f} seconds")
            print(f"Computation time: {computation_time:.6f} seconds")
    else:
        # Simulate GPU with optimized NumPy (will be slower than real GPU)
        C_cpu = np.dot(A_cpu, B_cpu)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Tiempo de ejecución GPU para N={N}: {execution_time:.6f} segundos")
    return execution_time

if __name__ == "__main__":
    main()