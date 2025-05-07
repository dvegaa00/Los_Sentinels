from project.processing.geo_utils import (read_shapefile, save_shapefile, convert_gdf_projection, 
                            fix_geometries, create_gdf_zones, calculate_areas, find_intersection)
from utils.visualization import plot_gdf_zones
import os
import tqdm
import geopandas as gpd

def process_intersection(scale):
    scale_path = os.path.join("./outputs", f"scale_{scale}")
    output_dir = f"/home/srodriguezr2/retina/outputs/interseccion/scale_{scale}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar todos los shapefiles
    gdfs = []
    for shapefile in tqdm.tqdm(os.listdir(scale_path), desc="Cargando shapefiles"):
        try:
            shp_path = os.path.join(scale_path, shapefile, f"Zonas_Priorizadas_{shapefile}.shp")
            gdf = read_shapefile(shp_path)
            gdfs.append(convert_gdf_projection(gdf))
        except Exception as e:
            print(f"❌ Error cargando {shapefile}: {e}")
    
    if not gdfs:
        raise ValueError("No se encontraron shapefiles para procesar.")
    
    # Realizar intersección
    gdf_intersected = gdfs[0]
    for gdf in tqdm.tqdm(gdfs[1:], desc="Calculando intersección"):
        gdf_intersected = find_intersection(fix_geometries(gdf_intersected), fix_geometries(gdf))
        
        
    
    # Procesar resultados
    gdf_processed = create_gdf_zones(gdf_intersected)
    gdf_processed = calculate_areas(gdf_processed)
    
    # Guardar resultados
    output_path = os.path.join(output_dir, f"Zonas_Priorizadas_interseccion_{scale}.shp")
    save_shapefile(gdf_processed, output_path)
    plot_gdf_zones(gdf_processed, os.path.join(output_dir, "Zonas_Priorizadas.png"))

def main():
    scales = ["10k", "25k"]
    for scale in scales:
        process_intersection(scale)

if __name__ == "__main__":
    main()