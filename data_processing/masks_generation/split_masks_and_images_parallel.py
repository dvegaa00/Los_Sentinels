import os
import rasterio
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

input_path = "/home/srodriguezr2/srodriguezr2_2/retina/Los_Sentinels/datasets/sentinel2_binary/complete"
output_path = "/home/srodriguezr2/srodriguezr2_2/retina/Los_Sentinels/datasets/sentinel2_binary/processed"

images_root = os.path.join(input_path, "images")
masks_root = os.path.join(input_path, "masks")

def split_array(arr):
    h, w = arr.shape[-2:]
    if w >= h:
        mid = w // 2
        return arr[..., :, :mid], arr[..., :, mid:]
    else:
        mid = h // 2
        return arr[..., :mid, :], arr[..., mid:, :]

def process_period(period_name):
    period_image_dir = os.path.join(images_root, period_name)
    period_mask_dir = os.path.join(masks_root, period_name)

    if not os.path.isdir(period_image_dir):
        return

    print(f"[{period_name}] Starting...")
    for filename in os.listdir(period_image_dir):
        if not filename.endswith(".tif") or not filename.startswith("zona_"):
            continue

        parts = filename.split("_")
        if len(parts) < 3:
            continue

        zone_id = parts[1]
        image_path = os.path.join(period_image_dir, filename)
        mask_path = os.path.join(period_mask_dir, f"zona_{zone_id}", "mascara_coberturas_agrupadas.tif")

        if not os.path.exists(mask_path):
            print(f"[{period_name}] Mask missing for zone {zone_id}")
            continue

        try:
            with rasterio.open(image_path) as img_src:
                img_data = img_src.read()
                img_meta = img_src.meta

            with rasterio.open(mask_path) as mask_src:
                mask_data = mask_src.read()
                mask_meta = mask_src.meta

            img_1, img_2 = split_array(img_data)
            mask_1, mask_2 = split_array(mask_data)

            for idx, (img_part, mask_part) in enumerate(zip([img_1, img_2], [mask_1, mask_2]), start=1):
                save_dir = os.path.join(output_path, period_name, f"zona_{zone_id}", f"part_{idx}")
                os.makedirs(save_dir, exist_ok=True)

                img_filename = f"{period_name}_zona_{zone_id}_image.tif"
                mask_filename = f"{period_name}_zona_{zone_id}_mask.tif"

                img_save_path = os.path.join(save_dir, img_filename)
                mask_save_path = os.path.join(save_dir, mask_filename)

                img_meta.update(height=img_part.shape[1], width=img_part.shape[2])
                mask_meta.update(height=mask_part.shape[1], width=mask_part.shape[2])

                with rasterio.open(img_save_path, 'w', **img_meta) as dst:
                    dst.write(img_part)

                with rasterio.open(mask_save_path, 'w', **mask_meta) as dst:
                    dst.write(mask_part)
        except Exception as e:
            print(f"[{period_name}] Error processing zone {zone_id}: {e}")

    print(f"[{period_name}] Done.")

if __name__ == "__main__":
    periods = [p for p in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, p))]
    
    # Use a pool of workers equal to the number of CPUs
    with Pool(processes=16) as pool:
        list(tqdm(pool.imap_unordered(process_period, periods), total=len(periods)))
