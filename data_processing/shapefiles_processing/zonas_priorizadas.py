from project.processing.geo_utils import read_shapefile, save_shapefile, create_gdf_zones, calculate_areas, find_intersection, convert_gdf_projection, fix_geometries, find_union
from utils.visualization import plot_gdf_zones
import os
import tqdm

def process_scale(scale):
    input_path = f"./data_{scale}"
    output_base = f"./outputs/scale_{scale}"
    
    for shapefile in tqdm.tqdm(os.listdir(input_path), desc=f"Procesando escala {scale}"):
        try:
            output_dir = os.path.join(output_base, f"Zonas_Priorizadas_{shapefile}")
            os.makedirs(output_dir, exist_ok=True)
            
            gdf = read_shapefile(os.path.join(input_path, shapefile, f"{shapefile}.shp"))
            gdf_processed = create_gdf_zones(gdf)
            gdf_processed = calculate_areas(gdf_processed)
            
            save_shapefile(gdf_processed, os.path.join(output_dir, f"Zonas_Priorizadas_{shapefile}.shp"))
            plot_gdf_zones(gdf_processed, os.path.join(output_dir, "Zonas_Priorizadas.png"))
            
        except Exception as e:
            print(f"❌ Error procesando {shapefile}: {e}")

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


#TODO: Arreglar el main para escoger qué se quiere hacer
def main():
    scales = ["10k", "25k"]
    for scale in scales:
        process_scale(scale)

if __name__ == "__main__":
    main()