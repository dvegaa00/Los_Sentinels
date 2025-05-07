from utils import (read_shapefile, save_shapefile, convert_gdf_projection,
                            create_gdf_zones, calculate_areas, apply_buffer, find_union)
from utils import plot_gdf_zones
import os
import tqdm

def process_union(scale):
    input_path = os.path.join("./outputs", f"scale_{scale}")
    output_dir = f"/home/srodriguezr2/retina/outputs/Union_{scale}"
    os.makedirs(output_dir, exist_ok=True)
    
    all_geometries = []
    for shapefile in tqdm.tqdm(os.listdir(input_path), desc="Cargando shapefiles"):
        try:
            shp_path = os.path.join(input_path, shapefile, f"{shapefile}.shp")
            all_geometries.append(read_shapefile(shp_path))
        except Exception as e:
            print(f"❌ Error procesando {shapefile}: {e}")
    
    if not all_geometries:
        raise ValueError("No se encontraron shapefiles para procesar.")
    
    # Procesar unión
    all_geometries = [convert_gdf_projection(gdf) for gdf in all_geometries]
    gdf_merged = create_gdf_zones(all_geometries[0])
    
    for gdf in tqdm.tqdm(all_geometries[1:], desc="Uniendo geometrías"):
        gdf_merged = find_union(convert_gdf_projection(gdf))
    
    gdf_processed = calculate_areas(gdf_merged)
    
    # Guardar resultados
    output_path = os.path.join(output_dir, "Zonas_Unificadas.shp")
    save_shapefile(gdf_processed, output_path)
    plot_gdf_zones(gdf_processed, os.path.join(output_dir, "Zonas_Priorizadas.png"))

def main():
    scales = ["25k"]  # Solo 25k para unión
    for scale in scales:
        process_union(scale)

if __name__ == "__main__":
    main()