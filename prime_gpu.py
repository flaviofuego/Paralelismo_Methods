#!/usr/bin/env python3
import numpy as np
import time
import sys
import math

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python prime_gpu.py D")
        sys.exit(1)
    
    try:
        D = int(sys.argv[1])
        if D <= 0:
            raise ValueError("D must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Define range for D-digit numbers
    start_range = 10**(D-1)
    end_range = 10**D - 1
    
    # Import GPU libraries - wrapped in try/except to handle missing dependencies
    try:
        import cupy as cp
        has_cupy = True
        has_numba = False
        print("Using CuPy for GPU acceleration")
    except ImportError:
        has_cupy = False
        try:
            from numba import cuda, jit
            has_numba = True
            print("Using Numba for GPU acceleration")
        except ImportError:
            has_numba = False
            print("Warning: Neither CuPy nor Numba found. Using CPU implementation.")
    
    # Record start time
    start_time = time.time()
    
    if has_cupy:
        try:
            # Create array of all numbers in range
            numbers = cp.arange(start_range, end_range + 1, dtype=cp.int64)
            
            # Create an array to mark primes (1 = prime, 0 = not prime)
            is_prime = cp.ones(end_range - start_range + 1, dtype=cp.int8)
            
            # Define CUDA kernel for prime checking
            # Adding necessary include directives for the CUDA code
            cuda_code = r'''
            #include <stdint.h>
            #include <math.h>
            
            extern "C" __global__
            void check_primes(const long long* numbers, int8_t* is_prime, int n) {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < n) {
                    long long num = numbers[idx];
                    
                    // Check if num is prime
                    if (num <= 1) {
                        is_prime[idx] = 0;
                        return;
                    }
                    if (num <= 3) {
                        is_prime[idx] = 1;
                        return;
                    }
                    if (num % 2 == 0 || num % 3 == 0) {
                        is_prime[idx] = 0;
                        return;
                    }
                    
                    long long sqrt_num = sqrt((double)num) + 1;
                    
                    for (long long i = 5; i <= sqrt_num; i += 6) {
                        if (num % i == 0 || num % (i + 2) == 0) {
                            is_prime[idx] = 0;
                            return;
                        }
                    }
                    
                    is_prime[idx] = 1;
                }
            }
            '''
            
            # Compile and launch kernel
            module = cp.RawModule(code=cuda_code)
            kernel = module.get_function('check_primes')
            threads_per_block = 256
            blocks_per_grid = (len(numbers) + threads_per_block - 1) // threads_per_block
            
            kernel((blocks_per_grid,), (threads_per_block,), (numbers, is_prime, len(numbers)))
            
            # Count primes
            count = int(cp.sum(is_prime).get())
            
        except Exception as e:
            print(f"CuPy GPU implementation failed: {e}")
            print("Falling back to CPU implementation")
            count = count_primes_cpu(start_range, end_range)
            
    elif has_numba:
        try:
            # Numba CUDA implementation
            
            # Define device function for primality test
            @cuda.jit(device=True)
            def is_prime_device(n):
                if n <= 1:
                    return False
                if n <= 3:
                    return True
                if n % 2 == 0 or n % 3 == 0:
                    return False
                
                sqrt_n = int(math.sqrt(n)) + 1
                
                for i in range(5, sqrt_n, 6):
                    if n % i == 0 or n % (i + 2) == 0:
                        return False
                
                return True
            
            # Define the kernel
            @cuda.jit
            def prime_kernel(start, results):
                i = cuda.grid(1)
                if i < len(results):
                    num = start + i
                    results[i] = 1 if is_prime_device(num) else 0
            
            # Prepare data
            range_size = end_range - start_range + 1
            results = np.zeros(range_size, dtype=np.int8)
            
            # Configure the grid
            threads_per_block = 256
            blocks = (range_size + threads_per_block - 1) // threads_per_block
            
            # Launch the kernel
            prime_kernel[blocks, threads_per_block](start_range, results)
            
            # Count primes
            count = np.sum(results)
            
        except Exception as e:
            print(f"Numba GPU implementation failed: {e}")
            print("Falling back to CPU implementation")
            count = count_primes_cpu(start_range, end_range)
    else:
        # CPU implementation
        count = count_primes_cpu(start_range, end_range)
    
    # Record end time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Verify with known results for small values of D
    expected_counts = {1: 4, 2: 21, 3: 143, 4: 1061, 5: 8363}
    if D in expected_counts:
        print(f"Expected count: {expected_counts[D]}, Actual count: {count}")
        if count != expected_counts[D]:
            print("Warning: Count does not match expected value!")
    
    print(f"Número de primos con {D} dígitos: {count}")
    print(f"Tiempo de ejecución GPU: {execution_time:.6f} segundos")
    return count, execution_time

def is_prime_cpu(n):
    """
    Determine if a number is prime using trial division.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    sqrt_n = int(math.sqrt(n)) + 1
    
    for i in range(5, sqrt_n, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    
    return True

def count_primes_cpu(start, end):
    """
    Count prime numbers in the given range using CPU.
    """
    count = 0
    for num in range(start, end + 1):
        if is_prime_cpu(num):
            count += 1
    return count

if __name__ == "__main__":
    main()