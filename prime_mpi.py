#!/usr/bin/env python3
from mpi4py import MPI
import time
import sys
import math

def is_prime(n):
    """
    Determine if a number is prime using trial division algorithm.
    
    Args:
        n: Number to check
    
    Returns:
        bool: True if n is prime, False otherwise
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Only need to check up to sqrt(n) for factors
    sqrt_n = int(math.sqrt(n)) + 1
    
    # Check potential prime factors of form 6k±1
    for i in range(5, sqrt_n, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    
    return True

def count_primes_in_range(start, end):
    """
    Count prime numbers in the given range [start, end].
    
    Args:
        start: Lower bound of range (inclusive)
        end: Upper bound of range (inclusive)
    
    Returns:
        int: Count of prime numbers in range
    """
    count = 0
    for num in range(start, end + 1):
        if is_prime(num):
            count += 1
    return count

def main():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Check command line arguments
    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: python prime_mpi.py D")
        MPI.Finalize()
        sys.exit(1)
    
    try:
        D = int(sys.argv[1])
        if D <= 0:
            raise ValueError("D must be positive")
    except ValueError as e:
        if rank == 0:
            print(f"Error: {e}")
        MPI.Finalize()
        sys.exit(1)
    
    # Define range for D-digit numbers
    start_range = 10**(D-1)
    end_range = 10**D - 1
    
    # Record start time (only on rank 0)
    if rank == 0:
        start_time = time.time()
    
    # Calculate numbers per process
    total_numbers = end_range - start_range + 1
    numbers_per_process = total_numbers // size
    remainder = total_numbers % size
    
    # Calculate this process's range
    if rank < remainder:
        my_count = numbers_per_process + 1
        my_start = start_range + rank * my_count
    else:
        my_count = numbers_per_process
        my_start = start_range + rank * my_count + remainder
    
    my_end = my_start + my_count - 1
    
    # Count primes in this process's range
    local_count = count_primes_in_range(my_start, my_end)
    
    # Reduce all local counts to get the total count
    total_count = comm.reduce(local_count, op=MPI.SUM, root=0)
    
    # Record end time and print results (only on rank 0)
    if rank == 0:
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify with known results for small values of D
        expected_counts = {1: 4, 2: 21, 3: 143, 4: 1061, 5: 8363}
        if D in expected_counts:
            print(f"Expected count: {expected_counts[D]}, Actual count: {total_count}")
            if total_count != expected_counts[D]:
                print("Warning: Count does not match expected value!")
        
        print(f"Número de primos con {D} dígitos: {total_count}")
        print(f"Tiempo de ejecución MPI con {size} procesos: {execution_time:.6f} segundos")
        return total_count, execution_time
    
    return None, None

if __name__ == "__main__":
    main()