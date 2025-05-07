import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio

class MaskSplitter:
    def __init__(self):
        self.coco_dict = {
            "images": [],
            "annotations_10000": [],
            "annotations_25000": [],
            "categories": [
                {"id": 1, "name": "Bosque", "supercategory": "none"},
                {"id": 2, "name": "No_Bosque", "supercategory": "none"}
            ],
            "multiclass_categorias": [
                {"id": 1, "name": "Arbustales", "supercategory": 2},
                {"id": 2, "name": "Bosque", "supercategory": 1},
                {"id": 3, "name": "Cultivos", "supercategory": 2},
                {"id": 4, "name": "Fragmentados", "supercategory": 2},
                {"id": 5, "name": "Herbazales", "supercategory": 2},
                {"id": 6, "name": "Pastizales", "supercategory": 2},
                {"id": 7, "name": "Superficies_de_agua", "supercategory": 2},
                {"id": 8, "name": "Territorios_artificiales", "supercategory": 2},
                {"id": 9, "name": "Tierras_degradadas", "supercategory": 2},
                {"id": 10, "name": "Vegetacion_secundaria", "supercategory": 1},
                {"id": 11, "name": "Areas_abiertas_con_poca_vegetacion", "supercategory": 2},
                {"id": 12, "name": "Vegetacion_secundaria", "supercategory": 2},
                {"id": 13, "name": "Areas_humedas", "supercategory": 2}
            ]
        }
        
        self.image_id = 1

    def check_image(self, tif_image):
        image_array = np.array(tif_image)
        values = np.unique(image_array)
        return values, image_array

    def count_pixel_percentages(self, image_array):
        total_elements = image_array.size
        unique, counts = np.unique(image_array, return_counts=True)
        percentages = {int(val): (count / total_elements) * 100 for val, count in zip(unique, counts)}
        return percentages

    def split_mask_and_save_coordinates(self, mask, year, semester, zona, tile_size=256, max_percentage=25):
        # Get height and width
        h, w = mask.shape
        print(f"Mask shape: {mask.shape}")
        
        # Crear lista para guardar las coordenadas
        coords = []
        # Recorrer en pasos de tile_size
        for y in tqdm(range(0, h, tile_size)):
            for x in tqdm(range(0, w, tile_size)):
                # Definir el recorte
                tile = mask[y:y+tile_size, x:x+tile_size]

                # Revisar que la imagen tenga la forma especificada
                if tile.shape == (tile_size,tile_size):
                    # Tomar los valores de la imagen
                    values, image_array = self.check_image(tile)

                    if len(values) > 1:
                        # Get the percentage of background
                        percentages = self.count_pixel_percentages(image_array=image_array)
                        
                        # Only save masks that have at least max_percentage values
                        if any(p < max_percentage for p in percentages.values()):
                            continue
                        else:
                            print(f"Mask found with percentages: {percentages}") 
                            # Guardar coordenadas 
                            tile_name = f"tile_{x}_{y}"
                            coords.append({"image": tile_name, "x": x, "y": y})
                
                # Increase image id
                self.image_id += 1
        # Create json folder
        json_path = "/home/dvegaa/ReTINA/Los_Sentinels/coordinates"
        os.makedirs(json_path, exist_ok=True)

        # Guardar coordenadas como CSV
        coords_df = pd.DataFrame(coords)
        coords_df.to_csv(os.path.join(json_path, f"{year}_{semester}_zona{zona}_tile_coordinates.csv"), index=False)
        print(f"Se guardaron los parches y coordenadas del año {year} y semestre {semester}")


if __name__ == "__main__":  

    zona = 1
    year = "2023"
    semester = "S1"
    clouds = "5"

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

    split_mask = MaskSplitter()
    split_mask.split_mask_and_save_coordinates(mask=aligned_mask, year=year, semester=semester, zona=zona, tile_size=256)