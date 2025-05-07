import os
import tifffile as tiff
import matplotlib.pyplot as plt
import geopandas as gpd
import sys
import pathlib
import matplotlib.patches as patches
import rasterio
from rasterio.transform import rowcol

retina_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(retina_path))
from utils import *

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

path_mask = "/home/srodriguezr2/srodriguezr2_2/retina/project/outputs/masks/Bosque_No_Bosque_2023_SI_Escala_1_10000/zona_1/mascara_coberturas_agrupadas.tif"
path_image = "/home/srodriguezr2/srodriguezr2_2/retina/sentinel_images/sentinel2/median/2023-SI/zona_1_clouds_5.tif"

epsilon = 1e-10
with rasterio.open(path_mask) as mask_ds, rasterio.open(path_image) as image_ds:
    aligned_mask = mask_ds.read(1).astype(np.float32)
    banda_4 = image_ds.read(3).astype(np.float32)
    banda_8 = image_ds.read(4).astype(np.float32)
    ndvi = (banda_8 - banda_4) / (banda_8 + banda_4 + epsilon)

#    aligned_mask = reproject_mask_to_match_image(mask_ds, ndvi_ds)

#capa = composite.normalizedDifference(["B8", "B4"])

class_1 = aligned_mask.copy()
class_1[class_1 != 1] = 0  # deja solo los 1

class_2 = aligned_mask.copy()
class_2[class_2 != 2] = 0  # deja solo los 2

save_path = os.path.join("/home/srodriguezr2/srodriguezr2_2/retina/visualizations_mascaras_nuevas", f"Zona_{str(zona)}")
os.makedirs(save_path, exist_ok=True)

# Show submask
print("Visualize mask")
plt.figure()
plt.imshow(aligned_mask, cmap='gray', vmin=0, vmax=2)  # You can change cmap depending on the type of image
plt.axis('off')  # Hide axes
plt.savefig(os.path.join(save_path, "mask_crop.png"), format='png', bbox_inches='tight', pad_inches=0)

# Show satellite image NDVI
print("Visualize image")
plt.figure()
plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
plt.axis('off')
plt.savefig(os.path.join(save_path, "satelite_image_ndvi.png"), bbox_inches='tight', pad_inches=0)

# Visualize alignment
print("Visualizing aligned mask and image")
plt.figure()
plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
plt.imshow(aligned_mask, cmap='gray', alpha=0.4, vmin=0, vmax=2)
plt.axis('off')
plt.savefig(os.path.join(save_path, "ndvi_mask_alineada.png"), bbox_inches='tight', pad_inches=0)

print("Visualizing aligned mask and image class 1")
plt.figure()
plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
mask1 = (class_1 == 1)
plt.imshow(np.ma.masked_where(~mask1, class_1), cmap='gray', alpha=0.5)
plt.axis('off')
plt.savefig(os.path.join(save_path, "ndvi_class1.png"), bbox_inches='tight', pad_inches=0)

print("Visualizing aligned mask and image class 2")
plt.figure()
plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
mask2 = (class_2 == 2)
plt.imshow(np.ma.masked_where(~mask2, class_2), cmap='gray', alpha=0.5)
plt.axis('off')
plt.savefig(os.path.join(save_path, "ndvi_class2.png"), bbox_inches='tight', pad_inches=0)