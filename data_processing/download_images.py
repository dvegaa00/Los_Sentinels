import os
import subprocess
import requests
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

import shutil


# 1. Autenticación
gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)

# 2. Función para descargar archivos desde Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# 3. Función para listar y descargar archivos
def download_drive_folder_recursively(folder_id, local_path):
    os.makedirs(local_path, exist_ok=True)

    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

    for file in file_list:
        title = file['title']
        file_id = file['id']
        mime = file['mimeType']

        if mime == 'application/vnd.google-apps.folder':  # If it's a folder
            print(f"📁 Subfolder: {title}")
            download_drive_folder_recursively(file_id, os.path.join(local_path, title))  # Recursively handle subfolders
        else:  # It's a file
            local_file_path = os.path.join(local_path, title)
            if os.path.exists(local_file_path):
                print(f"✅ Skipping (already downloaded): {local_file_path}")
                continue

            print(f"⬇️ Downloading {title} to {local_file_path}")
            try:
                file.GetContentFile(local_file_path)
            except Exception as e:
                print(f"❌ Failed to download {title}: {e}")

def reorganize_sentinel_folders(parent_dir, output_dir):
    # Listar todos los subdirectorios en la carpeta principal
    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    parent_dir_suffix = parent_dir.split("/")[-1]
    parent_dir_file_suffix = f"{parent_dir_suffix}_"
    satellite_name, file_image_method = parent_dir_suffix.split("_")

    output_dir = f"{output_dir}/{satellite_name}/{file_image_method}"

    # Filtrar solo los directorios que siguen el patrón de parent_dir_file_suffix
    sentinel_dirs = [d for d in subdirs if d.startswith(parent_dir_file_suffix)]
    
    for dir_name in sentinel_dirs:
        # Ruta completa del directorio original
        original_path = os.path.join(parent_dir, dir_name)
        
        # Extraer el año-temporada (ejemplo: 2018-SII)
        year_season = dir_name.replace(parent_dir_file_suffix, "")
        
        # Dividir para obtener la temporada
        parts = year_season.split("-")
        if len(parts) == 2:
            year, season = parts
            
            # Crear directorio de temporada si no existe
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Definir la nueva ruta: /output_dir/image_method/year-season
            new_path = os.path.join(output_dir, year_season)
            
            # Mover el directorio
            shutil.move(original_path, new_path)
            print(f"Movido: {original_path} → {new_path}")
        else:
            print(f"Formato no reconocido: {dir_name}")
    
    print("Reorganización completada.")

# 4. ID de la carpeta principal
ROOT_FOLDER_ID = "1ggNpvEchPnRr_IlszeEE8LcG3TMQXSOE"
OUTPUT_DIR = "/home/srodriguezr2/srodriguezr2_2/retina/sentinel2_median"

# 5. Iniciar la descarga
download_drive_folder_recursively(ROOT_FOLDER_ID, OUTPUT_DIR)

IMAGES_OUTPUT_DIR = "/home/srodriguezr2/srodriguezr2_2/retina/data"

# 6. Organizar los folders
reorganize_sentinel_folders(OUTPUT_DIR, IMAGES_OUTPUT_DIR)
