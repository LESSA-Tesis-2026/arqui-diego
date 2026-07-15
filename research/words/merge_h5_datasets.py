"""Fusiona dos sesiones de captura H5 de la misma etiqueta de palabra/frase.

Este script es intencionalmente manual: edite los nombres de los archivos de entrada en el bloque principal
antes de ejecutarlo para evitar fusiones accidentales entre etiquetas distintas.
"""

import os

import h5py
import numpy as np

DATASET_MERGE_FOLDER = "data_to_merge"


def merge_h5_files(target_file_name, donor_file_name):
    """Agrega los datasets de muestra del H5 donante a un archivo H5 objetivo sin sobrescribir las claves de muestra."""
    target_path = os.path.join(DATASET_MERGE_FOLDER, target_file_name)
    donor_path = os.path.join(DATASET_MERGE_FOLDER, donor_file_name)

    if not os.path.exists(target_path):
        print(f"[ERROR] Target file was not found: {target_path}")
        return
    if not os.path.exists(donor_path):
        print(f"[ERROR] Donor file was not found: {donor_path}")
        return

    print("\n--- STARTING H5 MERGE ---")
    print(f"Target: {target_file_name}")
    print(f"Donor:  {donor_file_name}")

    with h5py.File(target_path, 'a') as target_file, h5py.File(donor_path, 'r') as donor_file:
        existing_sample_count = len(target_file.keys())
        donor_sample_count = len(donor_file.keys())

        print(f"[*] Existing target samples: {existing_sample_count}")
        print(f"[*] Donor samples to import: {donor_sample_count}")

        for offset, key in enumerate(donor_file.keys()):
            new_sample_index = existing_sample_count + offset
            new_dataset_name = f"sample_{new_sample_index}"
            keypoint_matrix = np.array(donor_file[key])
            target_file.create_dataset(new_dataset_name, data=keypoint_matrix)

        print(
            f"[SUCCESS] Merge complete. '{target_file_name}' now contains "
            f"{len(target_file.keys())} total samples."
        )


if __name__ == "__main__":
    # Cambie estos nombres de archivo al fusionar manualmente dos capturas H5 de la misma etiqueta.
    target_file = "hola.h5"
    donor_file = "hola2.h5"

    merge_h5_files(target_file, donor_file)
