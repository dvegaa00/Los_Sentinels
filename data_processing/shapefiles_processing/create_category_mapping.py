
from utils import (read_shapefile)
from tqdm import tqdm
import os
import json

def save_json(json_data, json_path):
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

def create_and_save_class_mapping(classes, json_path):

    unique_cls = [cls_name for cls_name in classes if cls_name not in [None, "SIN INFORMACION"]]

    # +1 para tener categoría fondo al rasterizar
    class_name2id = {cls_name:cls_id + 1 for cls_id, cls_name in enumerate(sorted(unique_cls))}

    save_json(class_name2id, json_path)


def main():

    scales = ['10k', '25k']
    

    for scale in scales:

        classes = []
        
        class_key = 'descripcio' if scale == '10k' else 'cob_agrup'
        data_root_path = f'/home/srodriguezr2/srodriguezr2_2/retina/data_{scale}/'

        for shp_date_take in tqdm(os.listdir(data_root_path), desc=f'Processing {scale}...'):
            input_shp = os.path.join(data_root_path, shp_date_take, f'{shp_date_take}.shp')

            print("Finding shapefile's intersection...")

            gdf = read_shapefile(input_shp)
            classes += list(gdf[class_key].unique())
        
        unique_classes = set(classes)
        create_and_save_class_mapping(unique_classes, f"/home/srodriguezr2/srodriguezr2_2/retina/project/outputs/category_mapping/mapping_{scale}.json")

        print(unique_classes)


        #Unique classes 10k: 
        #                    {'BOSQUE', None, 'SIN INFORMACION', 'NO BOSQUE'} TODO: Eliminar None y SIN INFORMACION
    

        #Unique classes 25k: 
        #                      {'Vegetación secundaria', 'Arbustales', 'Superficies de agua', 'Ã\x81reas abiertas con poca vegetaciÃ³n', 
        #                       'Territorios artificializados', 'VegetaciÃ³n secundaria', 'Bosques', 'Áreas húmedas', 'Fragmentados', 'Pastizales', 
        #                       'Tierras degradadas', 'Cultivos', 'Herbazales', 'Ã\x81reas hÃºmedas', 'Vegetaciòn secundaria', 'Áreas abiertas con poca vegetación'}


if __name__ == '__main__':

    main()