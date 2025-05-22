
from utils import (crear_mascara_coberturas_with_image, find_intersection, fix_geometries, read_shapefile, convert_gdf_projection)
from concurrent.futures import ProcessPoolExecutor
import os
import json
import glob
from tqdm import tqdm

def process_shp_date_take(shp_date_take, scale, amazon_crs, scales_intersection, all_zones, class_mapping, class_key, binarize, input_images_path, output_masks_path):
    data_root_path = f'/home/srodriguezr2/srodriguezr2_2/retina/data_{scale}/'
    input_shp = os.path.join(data_root_path, shp_date_take, f'{shp_date_take}.shp')

    take_date = shp_date_take.split("_")
    take_date = "-".join([take_date[-5], take_date[-4]])

    print(f"[{shp_date_take}] Finding shapefile's intersection...")
    breakpoint()
    gdf = read_shapefile(input_shp)
    gdf = convert_gdf_projection(gdf, amazon_crs)
    gdf = find_intersection(gdf, scales_intersection)
    gdf = fix_geometries(gdf)

    for zona_id in all_zones:
        zone_image_path = glob.glob(f"{input_images_path}/{take_date}/zona_{zona_id}_*.tif")

        if len(zone_image_path) == 0:
            print(f"[{take_date}] No image for zona {zona_id}. Invalid path .{input_images_path}/{take_date}")
            continue

        zone_image_path = zone_image_path[0]

        output_dir = f'{output_masks_path}/{take_date}/zona_{zona_id}'
        os.makedirs(output_dir, exist_ok=True)

        print(f"[{shp_date_take}] Creating mask for zona {zona_id}...")

        crear_mascara_coberturas_with_image(
            gdf=gdf,
            class_mapping=class_mapping,
            zone_image_path=zone_image_path,
            ruta_mascara_salida=os.path.join(output_dir, 'mascara_coberturas_agrupadas.tif'),
            ruta_visualizacion=os.path.join(output_dir, 'mascara_coberturas_agrupadas_visualizacion.png'),
            class_key=class_key,
            binarize=binarize
        )


def main():
    amazon_crs = 'epsg:3117'
    scales = ['10k']
    binarize = False

    instersection_all_zones_25k = fix_geometries(read_shapefile("/home/srodriguezr2/srodriguezr2_2/retina/outputs/interseccion/scale_25k/Zonas_Priorizadas_interseccion_25k.shp"))
    instersection_all_zones_10k = fix_geometries(read_shapefile("/home/srodriguezr2/srodriguezr2_2/retina/outputs/interseccion/scale_10k/Zonas_Priorizadas_interseccion_10k.shp"))

    scales_intersection = fix_geometries(find_intersection(instersection_all_zones_10k, instersection_all_zones_25k))
    scales_intersection = convert_gdf_projection(scales_intersection, amazon_crs)

    all_zones = instersection_all_zones_25k["zona_id"].tolist()

    for scale in scales:
        data_root_path = f"/home/srodriguezr2/srodriguezr2_2/retina/data_{scale}/"
        shp_date_takes = sorted(os.listdir(data_root_path))

        input_images_path = "/home/srodriguezr2/srodriguezr2_2/retina/data/sentinel_images/sentinel2/median"
        output_masks_path = "/home/srodriguezr2/srodriguezr2_2/retina/data/sentinel_masks/sentinel2/median"

        class_mapping = json.load(open(f"/home/srodriguezr2/srodriguezr2_2/retina/project/outputs/category_mapping/mapping_{scale}.json"))
        class_key = 'descripcio' if scale == '10k' else 'cob_agrup'

        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = []
            for shp_date_take in shp_date_takes:
                futures.append(
                    executor.submit(
                        process_shp_date_take,
                        shp_date_take,
                        scale,
                        amazon_crs,
                        scales_intersection,
                        all_zones,
                        class_mapping,
                        class_key,
                        binarize,
                        input_images_path,
                        output_masks_path
                    )
                )

            # Optional: wait for all to complete with progress bar
            for f in tqdm(futures, desc=f"Processing {scale}..."):
                f.result()


if __name__ == '__main__':
    main()