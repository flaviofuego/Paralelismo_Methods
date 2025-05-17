import os
import sys
import subprocess
import urllib.request
import platform
import winreg
import ctypes
import time

def is_admin():
    """Verifica si el script se está ejecutando con privilegios de administrador"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def download_file(url, filename):
    """Descarga un archivo desde una URL"""
    print(f"Descargando {filename} desde {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Descarga de {filename} completada.")
        return True
    except Exception as e:
        print(f"Error al descargar {filename}: {e}")
        return False

def install_msmpi():
    """Descarga e instala Microsoft MPI"""
    # URLs de descarga para MSMPI
    msmpi_runtime_url = "https://download.microsoft.com/download/a/5/2/a5207ca5-1203-491a-8fb8-906fd68ae623/msmpisetup.exe"
    msmpi_sdk_url = "https://download.microsoft.com/download/a/5/2/a5207ca5-1203-491a-8fb8-906fd68ae623/msmpisdk.msi"
    
    # Nombres de los archivos
    msmpi_runtime_file = "msmpisetup.exe"
    msmpi_sdk_file = "msmpisdk.msi"
    
    # Descargar los archivos
    if not download_file(msmpi_runtime_url, msmpi_runtime_file):
        return False
    if not download_file(msmpi_sdk_url, msmpi_sdk_file):
        return False

    # Función para ejecutar como administrador
    def run_as_admin(cmd, args=None):
        if args is None:
            args = []
        
        try:
            # Usar ShellExecute para elevar privilegios
            print(f"Ejecutando '{cmd}' con privilegios de administrador...")
            if isinstance(args, list):
                args_str = ' '.join(args)
            else:
                args_str = args
            
            ctypes.windll.shell32.ShellExecuteW(
                None,                  # HWND parent
                "runas",               # Operación ("runas" significa "ejecutar como administrador")
                cmd,                   # Archivo a ejecutar
                args_str,              # Parámetros
                None,                  # Directorio
                1                      # SW_SHOWNORMAL
            )
            
            # Dar tiempo para completar la instalación
            print("Esperando a que termine la instalación...")
            time.sleep(40)  # Aumentamos el tiempo de espera para instalaciones más grandes
            
            return True
        except Exception as e:
            print(f"Error al ejecutar como administrador: {e}")
            return False
    
    # Instalar el runtime
    print("Instalando MSMPI Runtime...")
    try:
        runtime_installed = run_as_admin(os.path.abspath(msmpi_runtime_file), "/install /passive")
        
        if not runtime_installed:
            print("No se pudo instalar automáticamente. Intenta manualmente:")
        else:
            print("Instalación de MSMPI Runtime completada.")
    except subprocess.CalledProcessError as e:
        print(f"Error al instalar MSMPI Runtime: {e}")
        return False
    
    # Instalar el SDK
    print("Instalando MSMPI SDK...")
    try:
        sdk_installed = run_as_admin("msiexec", f"/i {os.path.abspath(msmpi_sdk_file)} /passive")
        
        if not sdk_installed:
            print("No se pudo instalar el SDK automáticamente. Intenta manualmente:")
        else:
            print("Instalación de MSMPI SDK completada.")
    except subprocess.CalledProcessError as e:
        print(f"Error al instalar MSMPI SDK: {e}")
        return False
    
    print("Esperando a que las instalaciones se completen...")
    time.sleep(40)  # Dar tiempo para que las instalaciones se completen

    return True

def set_environment_variables():
    """Configura las variables de entorno necesarias para MPI"""
    try:
        print("Buscando la instalación de MSMPI...")
        # Rutas típicas de instalación de MSMPI
        possible_msmpi_paths = [
            f"{os.environ.get('SYSTEMDRIVE', 'C:')}\\Program Files\\Microsoft MPI",
            f"{os.environ.get('SYSTEMDRIVE', 'C:')}\\Program Files (x86)\\Microsoft MPI"
        ]
        
        msmpi_path = None
        for path in possible_msmpi_paths:
            if os.path.exists(path):
                msmpi_path = path
                print(f"¡Encontrada instalación de MSMPI en {msmpi_path}!")
                break
        
        if not msmpi_path:
            print("No se pudo encontrar la instalación de MSMPI.")
            print("Por favor, verifica que Microsoft MPI esté instalado correctamente.")
            return False
        
        # Añadir a PATH la ubicación del binario de MPI
        msmpi_bin = os.path.join(msmpi_path, "Bin")
        
        # Verificar SDK
        possible_sdk_paths = [
            f"{os.environ.get('SYSTEMDRIVE', 'C:')}\\Program Files\\Microsoft SDKs\\MPI", 
            f"{os.environ.get('SYSTEMDRIVE', 'C:')}\\Program Files (x86)\\Microsoft SDKs\\MPI"
        ]
        
        sdk_path = None
        for path in possible_sdk_paths:
            if os.path.exists(path):
                sdk_path = path
                print(f"¡Encontrada instalación del SDK en {sdk_path}!")
                break
        
        if not sdk_path:
            print("No se pudo encontrar la instalación del SDK de MSMPI.")
            return False
        
        # Añadir variables de entorno
        env_vars = {
            "MSMPI_BIN": msmpi_bin,
            "MSMPI_INC": os.path.join(sdk_path, "Include"),
            "MSMPI_LIB32": os.path.join(sdk_path, "Lib\\x86"),
            "MSMPI_LIB64": os.path.join(sdk_path, "Lib\\x64")
        }
        
        print("Estableciendo variables de entorno...")
        
        try:
            # Actualizar PATH para el usuario actual (menos problemático que a nivel de sistema)
            path_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_ALL_ACCESS)
            try:
                path_value, _ = winreg.QueryValueEx(path_key, "Path")
            except FileNotFoundError:
                path_value = ""
                
            # Añadir MSMPI_BIN si no está ya
            if msmpi_bin.lower() not in path_value.lower():
                new_path = f"{path_value};{msmpi_bin}" if path_value else msmpi_bin
                winreg.SetValueEx(path_key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
                print(f"Añadido {msmpi_bin} a PATH del usuario")
            
            winreg.CloseKey(path_key)
            
            # También intentamos actualizar PATH del sistema
            try:
                system_path_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", 0, winreg.KEY_ALL_ACCESS)
                system_path_value, _ = winreg.QueryValueEx(system_path_key, "Path")
                
                if msmpi_bin.lower() not in system_path_value.lower():
                    new_system_path = f"{system_path_value};{msmpi_bin}"
                    winreg.SetValueEx(system_path_key, "Path", 0, winreg.REG_EXPAND_SZ, new_system_path)
                    print(f"Añadido {msmpi_bin} a PATH del sistema")
                
                winreg.CloseKey(system_path_key)
            except Exception as e:
                print(f"No se pudo actualizar PATH del sistema: {e}")
                print("Se ha configurado solo el PATH del usuario.")
        except Exception as e:
            print(f"Error al actualizar PATH: {e}")
        
        # Establecer otras variables de entorno (primero a nivel de usuario)
        for name, value in env_vars.items():
            try:
                user_env_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_ALL_ACCESS)
                winreg.SetValueEx(user_env_key, name, 0, winreg.REG_SZ, value)
                winreg.CloseKey(user_env_key)
                print(f"Variable de entorno {name} establecida a {value} para el usuario")
                
                # Actualizar también en la sesión actual
                os.environ[name] = value
            except Exception as e:
                print(f"Error al establecer la variable de entorno {name} para el usuario: {e}")
        
        # Notificar al sistema que las variables de entorno han cambiado
        try:
            # Enviar mensaje WM_SETTINGCHANGE para actualizar las variables de entorno
            HWND_BROADCAST = 0xFFFF
            WM_SETTINGCHANGE = 0x001A
            SMTO_ABORTIFHUNG = 0x0002
            result = ctypes.c_long()
            ctypes.windll.user32.SendMessageTimeoutW(
                HWND_BROADCAST, WM_SETTINGCHANGE, 0, 
                "Environment", SMTO_ABORTIFHUNG, 5000, ctypes.byref(result)
            )
            print("Notificado al sistema sobre el cambio en las variables de entorno.")
        except Exception as e:
            print(f"Error al notificar cambios de variables de entorno: {e}")
        
        print("Variables de entorno configuradas correctamente.")
        print("IMPORTANTE: Es posible que necesites reiniciar las ventanas de comando o tu IDE para que los cambios surtan efecto.")
        return True
    except Exception as e:
        print(f"Error al configurar las variables de entorno: {e}")
        return False


def install_mpi4py():
    """Instala mpi4py usando pip"""
    print("Instalando mpi4py...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "mpi4py"], check=True)
        print("mpi4py instalado correctamente.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error al instalar mpi4py: {e}")
        return False

def main():
    """Función principal"""
    print("=== Script de instalación y configuración de MPI en Windows ===")
    
    # Verificar privilegios de administrador
    if not is_admin():
        print("Este script requiere privilegios de administrador.")
        print("Por favor, ejecuta el script como administrador.")
        input("Presiona Enter para salir...")
        return
    
    # Verificar sistema operativo
    if platform.system() != "Windows":
        print("Este script solo funciona en Windows.")
        return
    
    # Instalar MSMPI
    if not install_msmpi():
        print("Error en la instalación de MSMPI. Abortando.")
        return
    
    # Configurar variables de entorno
    if not set_environment_variables():
        print("Error al configurar las variables de entorno. Es posible que algunas características no funcionen correctamente.")
    
    # Instalar mpi4py
    """ if not install_mpi4py():
        print("Error al instalar mpi4py. Abortando.")
        return """
    
    print("\nPara ejecutar programas MPI, usa:")
    print("mpiexec -n <num_procesos> python tu_script_mpi.py")

if __name__ == "__main__":
    main()