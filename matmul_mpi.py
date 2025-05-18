#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import time
import sys

def main():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Check command line arguments
    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: python matmul_mpi.py N")
        MPI.Finalize()
        sys.exit(1)
    
    try:
        N = int(sys.argv[1])
        if N <= 0:
            raise ValueError("N must be positive")
    except ValueError as e:
        if rank == 0:
            print(f"Error: {e}")
        MPI.Finalize()
        sys.exit(1)
    
    # Only rank 0 generates matrices and distributes
    if rank == 0:
        # Generate random matrices
        np.random.seed(42)  # For reproducibility
        A = np.random.random((N, N))
        B = np.random.random((N, N))
        
        # Record start time
        start_time = time.perf_counter()
    else:
        A = None
        B = None
    
    # Broadcast matrix B to all processes since it's needed in full by each process
    B = comm.bcast(B, root=0)
    
    # Calculate rows per process - distribute the work
    rows_per_process = N // size
    remainder = N % size
    
    # Adjust for uneven divisions
    if rank < remainder:
        my_rows = rows_per_process + 1
        offset = rank * my_rows
    else:
        my_rows = rows_per_process
        offset = rank * my_rows + remainder
        
    # Rank 0 distributes slices of A
    if rank == 0:
        local_A = A[:my_rows]
        
        # Send slices to other processes
        for r in range(1, size):
            # Calculate rows for process r
            if r < remainder:
                r_rows = rows_per_process + 1
                r_offset = r * r_rows
            else:
                r_rows = rows_per_process
                r_offset = r * r_rows + remainder
                
            comm.Send(A[r_offset:r_offset+r_rows], dest=r)
    else:
        # Other processes receive their slice
        local_A = np.empty((my_rows, N), dtype=np.float64)
        comm.Recv(local_A, source=0)
    
    # Each process performs matrix multiplication on its slice
    local_C = np.zeros((my_rows, N), dtype=np.float64)
    for i in range(my_rows):
        for j in range(N):
            for k in range(N):
                local_C[i, j] += local_A[i, k] * B[k, j]
    
    # Gather results back to rank 0
    if rank == 0:
        # Initialize the result matrix
        C = np.zeros((N, N), dtype=np.float64)
        
        # Copy local result to final matrix
        C[:my_rows] = local_C
        
        # Receive results from other processes
        for r in range(1, size):
            # Calculate rows for process r
            if r < remainder:
                r_rows = rows_per_process + 1
                r_offset = r * r_rows
            else:
                r_rows = rows_per_process
                r_offset = r * r_rows + remainder
                
            recv_buffer = np.empty((r_rows, N), dtype=np.float64)
            comm.Recv(recv_buffer, source=r)
            C[r_offset:r_offset+r_rows] = recv_buffer
        
        # Record end time and calculate execution time
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Verify result for small matrices (optional validation)
        if N <= 10:
            numpy_result = np.dot(A, B)
            error = np.max(np.abs(C - numpy_result))
            print(f"Maximum error compared to NumPy: {error:.6e}")
        
        print(f"Tiempo de ejecución MPI para N={N} con {size} procesos: {execution_time:.6f} segundos")
    else:
        # Send local result back to rank 0
        comm.Send(local_C, dest=0)

if __name__ == "__main__":
    main()