import os
import matplotlib.pyplot as plt
import rasterio
import numpy as np

zona = 8
year = "2023"
semester = "S1"
clouds = "70"

if semester == "S1":
    path_mask = f"/home/dvegaa/ReTINA/sentinel_masks/sentinel2/median/{year}-SI/zona_{str(zona)}/mascara_coberturas_agrupadas.tif"
    path_image = f"/home/dvegaa/ReTINA/sentinel_images/sentinel2/median/{year}-SI/zona_{str(zona)}_clouds_{clouds}.tif"
else:
    path_mask = f"/home/dvegaa/ReTINA/sentinel_masks/sentinel2/median/{year}-SI/zona_{str(zona)}/mascara_coberturas_agrupadas.tif"
    path_image = f"/home/dvegaa/ReTINA/sentinel_images/sentinel2/median/{year}-SII/zona_{str(zona)}_clouds_{clouds}.tif"

epsilon = 1e-10
with rasterio.open(path_mask) as mask_ds, rasterio.open(path_image) as image_ds:
    aligned_mask = mask_ds.read(1).astype(np.float32)
    banda_4 = image_ds.read(3).astype(np.float32)
    banda_8 = image_ds.read(4).astype(np.float32)
    ndvi = (banda_8 - banda_4) / (banda_8 + banda_4 + epsilon)


with rasterio.open(path_image) as image_ds:
    # Lee las 3 bandas (r, g, b)
    r = image_ds.read(1).astype(np.float32)
    g = image_ds.read(2).astype(np.float32)
    b = image_ds.read(3).astype(np.float32)
    
    # Combina las bandas en una sola imagen (H, W, 3)
    rgb = np.stack([r, g, b], axis=-1)

    # Normaliza para visualizar correctamente
    max_rgb = np.nanmax(rgb)
    rgb /= max_rgb

save_path = os.path.join("/home/dvegaa/ReTINA/Los_Sentinels/Results", f"{year}_{semester}", f"Zona_{str(zona)}")
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