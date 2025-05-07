from utils import read_shapefile, convert_gdf_projection, fix_geometries, limpiar_texto
import geopandas as gpd
import pandas as pd
import json
import os
import tqdm

def compute_class_areas(gdf):
    """Calcula áreas por clase, superclase y condición."""
    condition_areas = {desc: gdf[gdf["condicion_x"] == desc]["area_km2_x"].sum()
                      for desc in gdf["condicion_x"].unique()}
    
    superclass_areas = {desc: gdf[gdf["cob_agrup_x"] == desc]["area_km2_x"].sum()
                       for desc in gdf["cob_agrup_x"].unique()}
    
    class_areas = {desc: gdf[gdf["cobertura_x"] == desc]["area_km2_x"].sum()
                   for desc in gdf["cobertura_x"].unique()}
    
    return condition_areas, superclass_areas, class_areas

def main():
    intersection_shp = "./outputs/interseccion_25k/Zonas_Priorizadas_Zonas_Priorizadas_Coberturas_de_la_Tierra_2021_SI_Escala_1_25000.shp"
    hierarchy = json.load(open("./files/subcategories.json"))
    
    gdf_intersection = fix_geometries(convert_gdf_projection(read_shapefile(intersection_shp)))
    
    for scale in ["25k"]:
        scale_path = os.path.join("./", f"data_{scale}")
        
        for shapefile in tqdm.tqdm(os.listdir(scale_path), desc="Procesando clases"):
            try:
                shp_path = os.path.join(scale_path, shapefile, f"{shapefile}.shp")
                gdf = fix_geometries(convert_gdf_projection(read_shapefile(shp_path)))
                
                # Intersección con zonas priorizadas
                gdf_intersected = gpd.overlay(gdf, gdf_intersection, how="intersection")
                gdf_intersected = gdf_intersected[~gdf_intersected["geometry"].is_empty]
                
                # Calcular áreas
                condition_areas, superclass_areas, class_areas = compute_class_areas(gdf_intersected)
                
                # Limpiar nombres y guardar CSVs
                output_dir = os.path.join("./outputs", f"scale_{scale}", f"Zonas_Priorizadas_{shapefile}")
                os.makedirs(output_dir, exist_ok=True)
                
                for data, filename in zip(
                    [condition_areas, superclass_areas, class_areas],
                    ["Area_condition_intersection.csv", "Area_superclasses_intersection.csv", "Area_classes_intersection.csv"]
                ):
                    clean_data = {limpiar_texto(k): v for k, v in data.items()}
                    pd.DataFrame(clean_data).to_csv(os.path.join(output_dir, filename))
                
            except Exception as e:
                print(f"❌ Error procesando {shapefile}: {e}")

if __name__ == "__main__":
    main()