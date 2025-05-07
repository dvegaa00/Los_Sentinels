import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils

from rasterio.features import rasterize
from skimage import measure
from shapely.geometry import box
from tqdm import tqdm
from skimage.transform import resize


def convert_gdf_projection(gdf, target_crs='epsg:3857'):
    """Convierte un GeoDataFrame a un CRS proyectado si no lo está."""
    #if gdf.crs is None or not gdf.crs.is_projected:
    gdf = gdf.to_crs(target_crs)
    return gdf

def read_shapefile(shapefile_path):
    """Lee un shapefile y devuelve un GeoDataFrame."""
    return gpd.read_file(shapefile_path)

def find_intersection(gdf1, gdf2):

    if 'area_km2' in gdf1.columns and 'area_km2' in gdf2.columns:
        gdf1 = gdf1.drop(columns=['area_km2'])

    return gpd.overlay(gdf1, gdf2, how='intersection')

def fix_geometries(gdf):
    """Corrige geometrías inválidas."""
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf[~gdf.is_empty & gdf.is_valid]

def visualize_tif_image(tif_path, output_path):
    """
    Esta función carga una imagen .tif, la procesa y la guarda como imagen visualizable.
    
    Parámetros:
    tif_path (str): Ruta del archivo .tif a procesar.
    output_path (str): Ruta donde se guardará la imagen visualizada.
    """
    # Abrir la imagen
    with rasterio.open(tif_path) as src:
        image = src.read()  # Lee todas las bandas
        print(f"Shape (bands, height, width): {image.shape}")

    # Procesar imagen RGB
    if image.shape[0] >= 3:
        rgb = image[:3].transpose(1, 2, 0).astype(np.float32)
        rgb /= 10000.0  # Normaliza
        rgb = np.clip(rgb, 0, 1)  # Evita valores fuera de rango
        rgb_small = resize(rgb, (rgb.shape[0]//10, rgb.shape[1]//10))
        plt.imsave(output_path, rgb_small)
    else:
        gray = image[0].astype(np.float32)
        gray /= gray.max() if gray.max() > 0 else 1
        plt.imsave(output_path, gray, cmap='gray')

    print(f"Imagen guardada en: {output_path}")

def crear_mascara_coberturas_with_image(gdf, class_mapping, zone_image_path,
                                        ruta_mascara_salida=None, ruta_visualizacion=None,
                                        class_key='cob_agrup', binarize=False):
    
    """Crea una máscara raster a partir de un shapefile y una imagen satelital."""

    def _binarize_mask(mask):
        MAPPING_25000_TO_10000 = {
            1: 2, 2: 1, 3: 2, 4: 2, 5: 2,
            6: 2, 7: 2, 8: 2, 9: 2, 10: 1,
            11: 1, 12: 1, 13: 2, 14: 2, 15: 2, 16: 2
        }
        mapped_mask = np.copy(mask)
        for old_val, new_val in MAPPING_25000_TO_10000.items():
            mapped_mask[mask == old_val] = new_val
        return (mapped_mask > 0).astype(np.uint8)

    # Visualización base
    if ruta_visualizacion:
        ruta_visualizacion_imagen = ruta_visualizacion.replace("mascara", "imagen")
        visualize_tif_image(zone_image_path, ruta_visualizacion_imagen)

    # Leer imagen para extraer metadatos
    with rasterio.open(zone_image_path) as src:
        img_crs = src.crs
        img_transform = src.transform
        img_width = src.width
        img_height = src.height
        img_bounds = src.bounds

    # Filtrar geometrías que intersecten el bounding box de la imagen
    bbox = box(*img_bounds)
    gdf = gdf.to_crs(img_crs)
    gdf_filtered = gdf[gdf.intersects(bbox)].copy()
    gdf_filtered = fix_geometries(gdf_filtered)
    
    if gdf_filtered.empty:
        print("⚠️ No se encontraron geometrías que intersecten con la imagen.")
        return np.zeros((img_height, img_width), dtype=np.uint8)

    # Intersección para recorte preciso
    gdf_filtered['geometry'] = gdf_filtered.geometry.intersection(bbox)
    gdf_filtered = fix_geometries(gdf_filtered)

    # Rasterización
    shapes = [
        (geom, class_mapping[val])
        for geom, val in zip(gdf_filtered.geometry, gdf_filtered[class_key])
        if val in class_mapping
    ]

    mask = rasterize(
        shapes=shapes,
        out_shape=(img_height, img_width),
        transform=img_transform,
        fill=0,
        dtype=np.uint8
    )
    
    if binarize:
        mask = _binarize_mask(mask)

    # Guardar la máscara
    if ruta_mascara_salida:
        with rasterio.open(ruta_mascara_salida, 'w', driver='GTiff',
                           height=img_height, width=img_width, count=1,
                           dtype=mask.dtype, crs=img_crs, transform=img_transform) as dst:
            dst.write(mask, 1)

    # Visualización
    if ruta_visualizacion:
        plt.imshow(mask, cmap='tab20b')
        plt.axis('off')
        plt.savefig(ruta_visualizacion, bbox_inches='tight', pad_inches=0)
        plt.close()

    return mask
