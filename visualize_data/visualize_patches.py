import os
import matplotlib.pyplot as plt
import rasterio
from tqdm import tqdm
import pandas as pd
from matplotlib.colors import ListedColormap
import numpy as np

zona = 1
year = "2023"
semester = "S1"
clouds = "5"

if semester == "S1":
    path_mask = f"/home/dvegaa/ReTINA/sentinel_masks/sentinel2/median/{year}-SI/zona_{str(zona)}/mascara_coberturas_agrupadas.tif"
    path_image = f"/home/dvegaa/ReTINA/sentinel_images/sentinel2/median/{year}-SI/zona_{str(zona)}_clouds_{clouds}.tif"
    #path_image = f"/home/dvegaa/ReTINA/sentinel_images_fake/zona{str(zona)}_2023_S1_RGB_FAKE.tif"
else:
    path_mask = f"/home/dvegaa/ReTINA/sentinel_masks/sentinel2/median/{year}-SI/zona_{str(zona)}/mascara_coberturas_agrupadas.tif"
    path_image = f"/home/dvegaa/ReTINA/sentinel_images/sentinel2/median/{year}-SII/zona_{str(zona)}_clouds_{clouds}.tif"


epsilon = 1e-10
with rasterio.open(path_mask) as mask_ds, rasterio.open(path_image) as image_ds:
    aligned_mask = mask_ds.read(1).astype(np.float32)
    banda_4 = image_ds.read(3).astype(np.float32)
    banda_8 = image_ds.read(4).astype(np.float32)
    ndvi = (banda_8 - banda_4) / (banda_8 + banda_4 + epsilon)


with rasterio.open(path_mask) as mask_ds, rasterio.open(path_image) as image_ds:
    # Lee las 3 bandas (r, g, b)
    aligned_mask = mask_ds.read(1).astype(np.float32)
    r = image_ds.read(1).astype(np.float32)
    g = image_ds.read(2).astype(np.float32)
    b = image_ds.read(3).astype(np.float32)
    
    # Combina las bandas en una sola imagen (H, W, 3)
    rgb = np.stack([r, g, b], axis=-1)

    # Normaliza para visualizar correctamente
    max_rgb = np.nanmax(rgb)
    rgb /= max_rgb


save_path = os.path.join("/home/dvegaa/ReTINA/Los_Sentinels/Results_fake", f"{year}_{semester}", f"Zona_{str(zona)}", "Patches")
os.makedirs(save_path, exist_ok=True)

######################################################################################################################################

# Ruta al archivo con parches
patches_csv_path = f"/home/dvegaa/ReTINA/Los_Sentinels/coordinates/{year}_{semester}_zona{str(zona)}_tile_coordinates.csv"
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