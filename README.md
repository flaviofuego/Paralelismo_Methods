# Parallel Computing Performance Analysis

This repository contains implementations and analysis tools for comparing the performance of sequential, MPI, and GPU-accelerated algorithms across two classic computational problems: matrix multiplication and prime number counting.

## Project Overview

This project explores parallel computing architectures through carefully designed implementations of two computational workloads:

1. **Matrix Multiplication**: Comparing the performance of multiplying two square matrices of size N×N using three different approaches.
2. **Prime Number Counting**: Measuring the efficiency of counting all prime numbers with exactly D digits across different parallel paradigms.

Each workload is implemented in three versions:
- Sequential Python
- MPI with mpi4py (distributed memory parallelism)
- GPU acceleration using CuPy/CUDA

The repository includes an interactive analysis application that generates visualizations to help understand performance characteristics, scalability, and efficiency of each implementation.

## Installation Requirements

### Basic Requirements
```bash
pip install numpy matplotlib pandas streamlit plotly
```

### MPI Support
```bash
# Windows
# Download and install Microsoft MPI from: https://www.microsoft.com/en-us/download/details.aspx?id=57467
pip install mpi4py

# Linux/macOS
sudo apt-get install libopenmpi-dev # Ubuntu/Debian
brew install open-mpi              # macOS with Homebrew
pip install mpi4py
```

### GPU Support
```bash
# For NVIDIA GPUs with CUDA support
pip install cupy-cuda11x  # Replace 11x with your CUDA version
# OR
pip install numba         # Alternative GPU acceleration
```

## Project Structure

```plaintext
.
├── app.py                 # Interactive Streamlit analysis application
├── matmul_seq.py          # Sequential matrix multiplication
├── matmul_mpi.py          # MPI-based matrix multiplication
├── matmul_gpu.py          # GPU-accelerated matrix multiplication
├── prime_seq.py           # Sequential prime counting
├── prime_mpi.py           # MPI-based prime counting
├── prime_gpu.py           # GPU-accelerated prime counting
└── results/               # Directory for storing performance results
    └── performance_data.json  # Performance measurement database
```

## Running the Implementations

### Matrix Multiplication

Sequential:
```bash
python matmul_seq.py N
# Example: python matmul_seq.py 1000
```

MPI:
```bash
mpiexec -n <num_processes> python matmul_mpi.py N
# Example: mpiexec -n 4 python matmul_mpi.py 1000
```

GPU:
```bash
python matmul_gpu.py N
# Example: python matmul_gpu.py 1000
```

### Prime Number Counting

Sequential:
```bash
python prime_seq.py D
# Example: python prime_seq.py 4
```

MPI:
```bash
mpiexec -n <num_processes> python prime_mpi.py D
# Example: mpiexec -n 4 python prime_mpi.py 4
```

GPU:
```bash
python prime_gpu.py D
# Example: python prime_gpu.py 4
```

## Analysis Application

The repository includes an interactive analysis application built with Streamlit that visualizes performance data and provides insights into scaling behavior.

Run the application with:
```bash
streamlit run app.py
```

The application offers:

1. **Test Execution**: Run benchmarks for selected implementations and parameters
2. **Result Visualization**: Generate comparative plots of execution times
3. **Scalability Analysis**: Examine how MPI performance scales with worker count
4. **Efficiency Metrics**: Calculate and visualize parallel efficiency
5. **Computational Complexity**: Analyze growth patterns in execution time

## Understanding the Algorithms

### Matrix Multiplication

The implementations follow standard matrix multiplication:
- **Sequential**: Triple-nested loops with O(N³) complexity
- **MPI**: Row-based distribution where each process computes a subset of the resulting matrix
- **GPU**: Utilizes massively parallel GPU cores for concurrent matrix operations

### Prime Number Counting

The implementations count primes in the range $[10^(D-1), 10^D - 1]$:
- **Sequential**: Trial division algorithm optimized with early cutoffs
- **MPI**: Range partitioning among processes for distributed counting
- **GPU**: Parallel primality testing across thousands of GPU threads

## Performance Characteristics

Typical performance patterns you may observe:

1. **Matrix Multiplication**:
   - Sequential performance grows with O(N³) complexity
   - MPI provides good speedup for moderate matrix sizes
   - GPU offers dramatic acceleration for large matrices

2. **Prime Number Counting**:
   - For small D values, sequential is often fastest due to minimal overhead
   - MPI provides moderate speedup for larger D values
   - GPU may struggle with small D values due to initialization overhead

## Analysis Methodology

The application calculates several key metrics:

1. **Speedup**: The ratio of sequential time to parallel time
2. **Efficiency**: Speedup divided by the number of processors
3. **Scalability**: How performance changes with increasing worker count


## Troubleshooting

### MPI Issues
- Ensure Microsoft MPI is properly installed (Windows) or Open MPI (Linux/macOS)
- Check that PATH environment variables are correctly set
- Try running with 1 process first to isolate MPI initialization problems

### GPU Issues
- Verify CUDA toolkit installation and compatible drivers
- Check cupy or numba version compatibility with your CUDA version
- The implementation includes fallback to CPU if GPU acceleration fails

### Application Issues
- For timing precision issues with very fast operations, try larger problem sizes
- If visualizations don't appear, check Streamlit version compatibility
