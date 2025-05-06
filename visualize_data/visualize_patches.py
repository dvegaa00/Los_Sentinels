import os
import matplotlib.pyplot as plt
import geopandas as gpd
import sys
import pathlib
import rasterio
from tqdm import tqdm
import pandas as pd
from matplotlib.colors import ListedColormap

retina_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(retina_path))
from scripts.utils import *

#Get bounds
path_shp = "/home/dvegaa/ReTINA/interseccion/scale_10k/Zonas_Priorizadas_interseccion_10k.shp"
gdf = gpd.read_file(path_shp)
gdf = gdf.to_crs("EPSG:4326")
bounds = gdf.total_bounds

#Define zones coordinates [xmin_lon, ymin_lat, xmax_lon, ymax_lat]
zonas = [
    {"id": 1, "bounds": [-74.72079653, -0.130331786, -73.99637514, 0.92532454]},
    {"id": 2, "bounds": [-75.30668917, 0.806811201, -74.97104808, 1.094386973]},
    {"id": 3, "bounds": [-76.39628739, 0.956933946, -76.07521922, 1.161263215]},
    {"id": 4, "bounds": [-76.05944061, 1.263586621, -75.87490682, 1.493635654]},
    {"id": 5, "bounds": [-74.40181057, 1.06107717, -73.78883587, 1.81448118]},
    {"id": 6, "bounds": [-73.42953349, 1.608678975, -71.78541715, 2.877770836]},
    {"id": 7, "bounds": [-73.84210418, 2.706450857, -73.56563893, 2.976003546]},
    {"id": 8, "bounds": [-73.9463501, 3.104388851, -73.79384612, 3.217615258]},
    {"id": 9, "bounds": [-68.10768821, 3.901274681, -67.70618035, 4.167607701]},
]

zona = 1
year = "2019"
semester = "S1"

if semester == "S1":
    path_mask = f"/home/dvegaa/ReTINA/masks_tif/Bosque_No_Bosque_{year}_SI_Escala_1_10000/mascara_coberturas_agrupadas.tif"
else:
    path_mask = f"/home/dvegaa/ReTINA/masks/Bosque_No_Bosque_{year}_SII_Escala_1_10000/mascara_coberturas_agrupadas.tif"

path_ndvi = f"/home/dvegaa/ReTINA/satellite_images/{year}_{semester}/zona{str(zona)}_{year}_{semester}_NDVI.tif"
path_rgb = f"/home/dvegaa/ReTINA/satellite_images/{year}_{semester}/zona{str(zona)}_{year}_{semester}_RGB.tif"


with rasterio.open(path_mask) as mask_ds, rasterio.open(path_ndvi) as ndvi_ds:
    ndvi = ndvi_ds.read(1).astype(np.float32)
    aligned_mask = reproject_mask_to_match_image(mask_ds, ndvi_ds)

save_path = os.path.join("/home/dvegaa/ReTINA/visualizations", f"{year}_{semester}", f"Zona_{str(zona)}", "Patches")
os.makedirs(save_path, exist_ok=True)

######################################################################################################################################

# Ruta al archivo con parches
patches_csv_path = f"/home/dvegaa/ReTINA/coordinates/{year}_{semester}_zona{str(zona)}_tile_coordinates.csv"
df_patches = pd.read_csv(patches_csv_path)
PATCH_SIZE = 256

for i, row in tqdm(df_patches.iterrows(), total=len(df_patches), desc="Visualizando parches"):
    x, y = int(row["x"]), int(row["y"])
    patch_id = row["image"]

    # Extraer el parche
    ndvi_patch = ndvi[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
    mask_patch = aligned_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

    # Visualizar y guardar NDVI
    plt.figure()
    plt.imshow(ndvi_patch, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f"patch_{patch_id}_ndvi.png"), bbox_inches='tight', pad_inches=0)
    
    # Visualizar y guardar mascara
    custom_cmap = ListedColormap(["black", "#1f6c07", "#82b772"])
    plt.figure()
    plt.imshow(mask_patch, cmap=custom_cmap, vmin=0, vmax=2)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f"patch_{patch_id}_mask.png"), bbox_inches='tight', pad_inches=0)

    # Visualizar y guardar NDVI + máscara
    plt.figure()
    plt.imshow(ndvi_patch, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.imshow(mask_patch, cmap='gray', alpha=0.4, vmin=0, vmax=2)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f"patch_{patch_id}_ndvi_mask.png"), bbox_inches='tight', pad_inches=0)