import geopandas as gpd
import re
from shapely.ops import unary_union
from shapely.validation import make_valid
import unicodedata

import matplotlib.pyplot as plt
import numpy as np

def read_shapefile(shapefile_path):
    """Lee un shapefile y devuelve un GeoDataFrame."""
    return gpd.read_file(shapefile_path)

def limpiar_texto(texto):
    """Normaliza y limpia texto eliminando tildes y corrigiendo errores comunes."""
    if not isinstance(texto, str):
        return texto
        
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8").lower()
    
    correcciones = {
        "vegetacia3n": "vegetacion",
        "arba3reos": "arboreos",
        "remocia3n": "remocion",
        "heteroganeo": "heterogeneo",
        "raos": "rios",
        "haomedos": "humedos",
        "haomedas": "humedas",
        "galeraa": "galeria",
        "mesa3filo": "mesofilo",
        "cianagas": "cienagas"
    }
    
    for error, correccion in correcciones.items():
        texto = re.sub(rf"\b{re.escape(error)}\b", correccion, texto)
    
    return texto

def convert_gdf_projection(gdf, target_crs='epsg:3117'):
    """Convierte un GeoDataFrame a un CRS proyectado si no lo está."""
    #if gdf.crs is None or not gdf.crs.is_projected:
    gdf = gdf.to_crs(target_crs)
    return gdf

def apply_buffer(gdf, buffer_size=0):
    """Aplica un buffer a las geometrías y corrige intersecciones."""
    gdf["geometry"] = gdf.geometry.buffer(buffer_size).buffer(0)
    return gdf

def fix_geometries(gdf):
    """Corrige geometrías inválidas."""
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf[~gdf.is_empty & gdf.is_valid]

def save_shapefile(gdf, output_path):
    """Guarda un GeoDataFrame como shapefile."""
    gdf.to_file(output_path)
    print(f"✅ Shapefile guardado en: {output_path}")

def create_gdf_zones(gdf, gdf_original=None, buffer_size=3000):
    """Crea zonas a partir de un GeoDataFrame."""
    if gdf_original is None:
        gdf_original = gdf.copy()
    
    gdf = convert_gdf_projection(gdf)
    gdf = apply_buffer(gdf, buffer_size)
    
    merged_geometry = unary_union(gdf.geometry)
    gdf_merged = gpd.GeoDataFrame(geometry=[merged_geometry], crs=gdf.crs)
    gdf_merged = gdf_merged.explode(index_parts=False)
    gdf_merged["zona_id"] = range(1, len(gdf_merged) + 1)
    
    gdf_original = gdf_original.to_crs(gdf_merged.crs)
    gdf_merged = gpd.sjoin(gdf_original, gdf_merged, how="left", predicate="intersects")
    gdf_merged["geometry"] = gdf_merged["geometry"].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
    gdf_merged = gdf_merged[~gdf_merged["geometry"].is_empty]
    
    return gdf_merged.dissolve(by="zona_id").reset_index()

def plot_gdf_zones(gdf, output_path, title="Zonas Priorizadas"):
    """Visualiza zonas geográficas con colores aleatorios."""
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = np.random.rand(len(gdf), 3)
    
    for i, (_, row) in enumerate(gdf.iterrows()):
        gpd.GeoSeries(row.geometry).plot(ax=ax, color=[colors[i]], alpha=0.7)
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, str(row["zona_id"]), 
               fontsize=8, ha="center", va="center", color="white",
               bbox=dict(facecolor="black", alpha=0.2))
    
    ax.set_title(f"{title} - {len(gdf)} zonas")
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_variacion_area(df, clase, save_path):
    """Genera gráfico de evolución temporal de áreas."""
    fig = px.line(df, x="periodo", y=clase, markers=True, 
                 title=f"Evolución del Área para {clase}")
    fig.write_image(save_path)