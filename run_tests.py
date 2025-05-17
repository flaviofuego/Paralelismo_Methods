import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de parámetros
matrix_sizes = [100, 200, 500, 1000, 2000]
digit_counts = [2, 3, 4]
mpi_workers = [1, 2, 4, 8, 16]

# Matrices para almacenar resultados
matrix_results = pd.DataFrame(columns=['N', 'Implementation', 'Workers', 'Time'])
prime_results = pd.DataFrame(columns=['D', 'Implementation', 'Workers', 'Time'])

# Ejecutar pruebas de multiplicación de matrices
for N in matrix_sizes:
    # Secuencial
    cmd = f"python matmul_seq.py  {N}"
    output = subprocess.check_output(cmd, shell=True).decode('cp1252')
    time_seq = float(output.split(":")[-1].strip().split()[0])
    new_row = pd.DataFrame({'N': [N], 'Implementation': ['Sequential'], 'Workers': [1], 'Time': [time_seq]})
    matrix_results = pd.concat([matrix_results, new_row], ignore_index=True)
    
    # MPI con diferentes números de trabajadores
    for workers in mpi_workers:
        cmd = f"mpiexec -n {workers} python matmul_mpi.py  {N}"
        output = subprocess.check_output(cmd, shell=True).decode('cp1252')
        time_mpi = float(output.split(":")[-1].strip().split()[0])
        new_row = pd.DataFrame({'N': [N], 'Implementation': ['MPI'], 'Workers': [workers], 'Time': [time_mpi]})
        matrix_results = pd.concat([matrix_results, new_row], ignore_index=True)
    
    # GPU
    cmd = f"python matmul_gpu  {N}"
    output = subprocess.check_output(cmd, shell=True).decode('cp1252')
    time_gpu = float(output.split(":")[-1].strip().split()[0])
    new_row = pd.DataFrame({'N': [N], 'Implementation': ['GPU'], 'Workers': [1], 'Time': [time_gpu]})
    matrix_results = pd.concat([matrix_results, new_row], ignore_index=True)

# Ejecutar pruebas de conteo de primos
for D in digit_counts:
    # Secuencial
    cmd = f"python prime_seq  {D}"
    output = subprocess.check_output(cmd, shell=True).decode('cp1252')
    time_seq = float(output.split("Tiempo de ejecución secuencial:")[-1].strip().split()[0])
    new_row = pd.DataFrame({'D': [D], 'Implementation': ['Sequential'], 'Workers': [1], 'Time': [time_seq]})
    prime_results = pd.concat([prime_results, new_row], ignore_index=True)
    
    # MPI con diferentes números de trabajadores
    for workers in mpi_workers:
        cmd = f"mpiexec -n {workers} python prime_mpi  {D}"
        output = subprocess.check_output(cmd, shell=True).decode('cp1252')
        time_mpi = float(output.split(":")[-1].strip().split()[0])
        new_row = pd.DataFrame({'D': [D], 'Implementation': ['MPI'], 'Workers': [workers], 'Time': [time_mpi]})
        prime_results = pd.concat([prime_results, new_row], ignore_index=True)
    
    # GPU
    cmd = f"python prime_gpu  {D}"
    output = subprocess.check_output(cmd, shell=True).decode('cp1252')
    time_gpu = float(output.split(":")[-1].strip().split()[0])
    new_row = pd.DataFrame({'D': [D], 'Implementation': ['GPU'], 'Workers': [1], 'Time': [time_gpu]})
    prime_results = pd.concat([prime_results, new_row], ignore_index=True)

# Guardar resultados
matrix_results.to_csv('matrix_results.csv', index=False)
prime_results.to_csv('prime_results.csv', index=False)

# Crear gráficas
# 1. Gráfica de tiempos MPI vs número de trabajadores para D=4
prime_d4 = prime_results[prime_results['D'] == 4]
prime_d4_mpi = prime_d4[prime_d4['Implementation'] == 'MPI']

plt.figure(figsize=(10, 6))
plt.plot(prime_d4_mpi['Workers'], prime_d4_mpi['Time'], marker='o')
plt.title('Tiempo de ejecución MPI vs Número de trabajadores (D=4)')
plt.xlabel('Número de trabajadores')
plt.ylabel('Tiempo (s)')
plt.grid(True)
plt.savefig('mpi_workers_time_d4.png')

# 2. Gráfica log-log de tiempo vs N para las tres implementaciones
plt.figure(figsize=(10, 6))
for impl in ['Sequential', 'MPI', 'GPU']:
    if impl == 'MPI':
        # Usar el número óptimo de trabajadores
        best_workers = matrix_results[matrix_results['Implementation'] == 'MPI'].groupby('N')['Time'].idxmin()
        best_data = matrix_results.loc[best_workers]
        plt.loglog(best_data['N'], best_data['Time'], marker='o', label=f'{impl} (mejor)')
    else:
        data = matrix_results[matrix_results['Implementation'] == impl]
        plt.loglog(data['N'], data['Time'], marker='o', label=impl)

plt.title('Tiempo de ejecución vs Tamaño de matriz (log-log)')
plt.xlabel('Tamaño de matriz (N)')
plt.ylabel('Tiempo (s)')
plt.legend()
plt.grid(True)
plt.savefig('matrix_time_comparison.png')

# 3. Gráfica log-log de tiempo vs D para las tres implementaciones
plt.figure(figsize=(10, 6))
for impl in ['Sequential', 'MPI', 'GPU']:
    if impl == 'MPI':
        # Usar el número óptimo de trabajadores
        best_workers = prime_results[prime_results['Implementation'] == 'MPI'].groupby('D')['Time'].idxmin()
        best_data = prime_results.loc[best_workers]
        plt.loglog(best_data['D'], best_data['Time'], marker='o', label=f'{impl} (mejor)')
    else:
        data = prime_results[prime_results['Implementation'] == impl]
        plt.loglog(data['D'], data['Time'], marker='o', label=impl)

plt.title('Tiempo de ejecución vs Número de dígitos (log-log)')
plt.xlabel('Número de dígitos (D)')
plt.ylabel('Tiempo (s)')
plt.legend()
plt.grid(True)
plt.savefig('prime_time_comparison.png')