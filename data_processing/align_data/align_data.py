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
from scripts.utils import *

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

class_1 = aligned_mask.copy()
class_1[class_1 != 1] = 0  # deja solo los 1

class_2 = aligned_mask.copy()
class_2[class_2 != 2] = 0  # deja solo los 2

with rasterio.open(path_rgb) as rgb_ds:
    # Lee las 3 bandas (r, g, b)
    r = rgb_ds.read(1).astype(np.float32)
    g = rgb_ds.read(2).astype(np.float32)
    b = rgb_ds.read(3).astype(np.float32)
    
    # Combina las bandas en una sola imagen (H, W, 3)
    rgb = np.stack([r, g, b], axis=-1)

    # Normaliza para visualizar correctamente
    max_rgb = np.nanmax(rgb)
    rgb /= max_rgb

save_path = os.path.join("/home/dvegaa/ReTINA/visualizations", f"{year}_{semester}", f"Zona_{str(zona)}")
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

# Show satellite image RGB
print("Visualize image")
plt.figure()
plt.imshow(rgb)
plt.axis('off')
plt.savefig(os.path.join(save_path, "satelite_image_rgb.png"), bbox_inches='tight', pad_inches=0)

# Visualize alignment
print("Visualizing aligned mask and image")
plt.figure()
plt.imshow(rgb, cmap='RdYlGn', vmin=-1, vmax=1)
plt.imshow(aligned_mask, cmap='gray', alpha=0.4, vmin=0, vmax=2)
plt.axis('off')
plt.savefig(os.path.join(save_path, "rgb_mask_alineada.png"), bbox_inches='tight', pad_inches=0)