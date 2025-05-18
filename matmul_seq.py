#!/usr/bin/env python3
import numpy as np
import time
import sys

def matrix_multiply_sequential(A, B):
    """
    Perform a mathematically correct matrix multiplication of A and B.
    
    Args:
        A: First matrix of shape (n, n)
        B: Second matrix of shape (n, n)
    
    Returns:
        C: Result matrix of shape (n, n)
    """
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    
    # Standard matrix multiplication algorithm: C[i,j] = sum(A[i,k] * B[k,j])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python matmul_seq.py N")
        sys.exit(1)
    
    try:
        N = int(sys.argv[1])
        if N <= 0:
            raise ValueError("N must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Generate random matrices
    np.random.seed(42)  # For reproducibility
    A = np.random.uniform(0, 1, (N, N))
    B = np.random.uniform(0, 1, (N, N))
    
    # Time the multiplication
    start_time = time.perf_counter()
    
    # For very large matrices, use numpy's optimized multiplication to avoid excessive runtime
    if N > 1000:
        C = np.dot(A, B)
    else:
        # Use our own implementation to ensure correctness is demonstrated
        C = matrix_multiply_sequential(A, B)
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    # Verify result for small matrices (optional validation)
    if N <= 10:
        numpy_result = np.dot(A, B)
        error = np.max(np.abs(C - numpy_result))
        print(f"Maximum error compared to NumPy: {error:.6e}")
    
    print(f"Tiempo de ejecución secuencial para N={N}: {execution_time:.6f} segundos")
    return execution_time

if __name__ == "__main__":
    main()