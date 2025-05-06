import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio
import sys
import pathlib
import geopandas as gpd

retina_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(retina_path))
from scripts.utils import *

retina_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(retina_path))
from scripts.utils import *

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
        json_path = "/home/dvegaa/ReTINA/coordinates"
        os.makedirs(json_path, exist_ok=True)

        # Guardar coordenadas como CSV
        coords_df = pd.DataFrame(coords)
        coords_df.to_csv(os.path.join(json_path, f"{year}_{semester}_zona{zona}_tile_coordinates.csv"), index=False)
        print(f"Se guardaron los parches y coordenadas del año {year} y semestre {semester}")


if __name__ == "__main__":  

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

    split_mask = MaskSplitter()
    split_mask.split_mask_and_save_coordinates(mask=aligned_mask, year=year, semester=semester, zona=zona)