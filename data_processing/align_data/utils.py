import numpy as np
import json
import os
from pycocotools import mask as maskUtils
from PIL import Image
import tifffile as tiff
from tqdm import tqdm
import rasterio
from rasterio.warp import reproject, Resampling

def get_coordinates_patch(mask, coordinates, tile_size=256):
    # Definir el tamaño de los recortes, que es fijo
    tile_size = 256  # Asegúrate de que es el tamaño correcto para los recortes

    # Obtener las coordenadas y el nombre del archivo de la fila
    x = coordinates[0]
    y = coordinates[1]

    # Generar el recorte usando las coordenadas
    cropped_tile = mask[y:y + tile_size, x:x + tile_size]

    return cropped_tile

########################################### GET COORDINATES #######################################
def lonlat_to_pixel(lon, lat, geo_bounds, width, height):
    """
    Convierte coordenadas geográficas (lon, lat) a coordenadas de píxeles (x, y)
    dentro de una imagen de dimensiones (width, height), dada su extensión geográfica.

    Parámetros:
        lon, lat: coordenadas en grados (longitud, latitud)
        geo_bounds: [xmin, ymin, xmax, ymax] → bounding box geográfico de la imagen
        width: ancho de la imagen (en píxeles)
        height: alto de la imagen (en píxeles)

    Retorna:
        (px, py): coordenadas de píxel (enteras)
    """
    xmin, ymin, xmax, ymax = geo_bounds

    # Ratios de posición dentro del rango geográfico
    x_ratio = (lon - xmin) / (xmax - xmin)
    y_ratio = (ymax - lat) / (ymax - ymin)  # invertido porque y=0 está arriba en imágenes

    # Convertir a coordenadas de píxel
    px = int(x_ratio * width)
    py = int(y_ratio * height)

    return px, py


def reproject_mask_to_match_image(mask, image):
    """
    Reproyecta una máscara rasterizada (abierta con rasterio) para que coincida
    en resolución, tamaño y CRS con una imagen de referencia (también abierta).

    Parámetros:
        mask_dataset (rasterio.DatasetReader): Raster de máscara ya abierto.
        image_dataset (rasterio.DatasetReader): Imagen base (NDVI) ya abierta.

    Retorna:
        np.ndarray: Máscara reproyectada (numpy array con shape de la imagen).
    """
    dst_crs = image.crs # Sistema de coordendas de la imagen
    dst_transform = image.transform # Tamaño de los pixeles y ubicacion de la imagen en el espacion geográfico
    dst_shape = (image.height, image.width) 

    src_data = mask.read(1) # Cargo la mascara 
    dst_data = np.empty(dst_shape, dtype=src_data.dtype) #Creo un array de tamaño de la imagen donde quedará la máscara correspondiente

    # Reproyectamos la mascara al espacion de la imagen 
    reproject(
        source=src_data, # Mascara original
        destination=dst_data, # Donde se va a guardar la máscata
        src_transform=mask.transform, # Ubicacion y resolucion de la máscara
        src_crs=mask.crs, # Sistema de coordenadas de la mascara
        dst_transform=dst_transform, # Ubicacion y resolucion de la imagen
        dst_crs=dst_crs, # Sistema de coordenadas de la imagen
        resampling=Resampling.nearest  # conservar clases
    )

    return dst_data