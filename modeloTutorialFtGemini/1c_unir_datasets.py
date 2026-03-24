import os
import h5py
import numpy as np

# Si tus archivos están en la carpeta "data", asegúrate de que la ruta sea correcta
CARPETA_DATOS = "data_unir" 

def fusionar_archivos_h5(nombre_principal, nombre_secundario):
    """
    Toma los datos de 'nombre_secundario' y los inyecta en 'nombre_principal',
    renombrando dinámicamente las llaves (samples) para evitar sobreescritura.
    """
    ruta_principal = os.path.join(CARPETA_DATOS, nombre_principal)
    ruta_secundaria = os.path.join(CARPETA_DATOS, nombre_secundario)

    # Validaciones de seguridad
    if not os.path.exists(ruta_principal):
        print(f"[ERROR] No se encontró el archivo principal: {ruta_principal}")
        return
    if not os.path.exists(ruta_secundaria):
        print(f"[ERROR] No se encontró el archivo a fusionar: {ruta_secundaria}")
        return

    print(f"\n--- INICIANDO FUSIÓN ---")
    print(f"Destino: {nombre_principal}")
    print(f"Origen:  {nombre_secundario}")

    # Abrimos el principal en modo 'a' (append) y el secundario en 'r' (read)
    with h5py.File(ruta_principal, 'a') as f_prin, h5py.File(ruta_secundaria, 'r') as f_sec:
        
        # Contamos cuántas muestras tiene ya el archivo principal para saber desde dónde empezar a numerar
        muestras_actuales = len(f_prin.keys())
        muestras_nuevas = len(f_sec.keys())
        
        print(f"[*] Muestras actuales en destino: {muestras_actuales}")
        print(f"[*] Muestras a importar: {muestras_nuevas}")
        
        for i, key in enumerate(f_sec.keys()):
            # Calculamos el nuevo nombre (Ej: si hay 80 muestras, el nuevo empieza en sample_80)
            nuevo_indice = muestras_actuales + i
            nuevo_nombre_dataset = f"sample_{nuevo_indice}"
            
            # Extraemos la matriz de coordenadas pura del archivo secundario
            datos_matematica = np.array(f_sec[key])
            
            # La inyectamos en el archivo principal
            f_prin.create_dataset(nuevo_nombre_dataset, data=datos_matematica)
            
        print(f"[ÉXITO] Fusión completada. '{nombre_principal}' ahora contiene {len(f_prin.keys())} muestras totales.")

if __name__ == "__main__":
    # --- ZONA DE EJECUCIÓN ---
    # Cambiar aquí el nombre exacto de tus archivos.
    # El primer parámetro es el archivo que va a RECIBIR los datos.
    # El segundo parámetro es el archivo que va a DONAR los datos.
    
    archivo_destino = "hola.h5"
    archivo_donante = "hola2.h5"
    
    fusionar_archivos_h5(archivo_destino, archivo_donante)