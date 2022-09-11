import argparse
import os
from pycocotools.coco import COCO
import numpy as np
from util import *
from tqdm import tqdm
import json


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test_case')
    parser.add_argument('--threshold', default=50, type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_config()
    threshold = args.threshold
    data_set = "COCO"
    data_dir = f"dataset/{data_set}"
    phase = "val"
    ann_file = f"{data_dir}/annotations/instances_{phase}2014.json"
    experiment_name = args.experiment_name
    
    coco = COCO(ann_file)

    cats = {k: v for k, v in sorted(coco.cats.items(), key=lambda cat: cat[1]['name'])}
    coco_id_to_new_id = {c:i for i, c in enumerate(cats)}

    predictions = []
    for img_id in tqdm(coco.imgs):
        img = coco.imgs[img_id]
        file_name = img['file_name']
        
        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(annIds)
        labels = list(set([coco_id_to_new_id[a["category_id"]] for a in anns]))

        for label in labels:
            scoremap_file = os.path.join("train_log", experiment_name, "scoremaps", phase, f"{file_name}_label_{label}.npy")
            scoremap = np.load(scoremap_file)
            centers = find_centers_of_mass_contours(scoremap, threshold=threshold)

            for center in centers:
                predictions.append({
                    "category_id": label,
                    "image_name": file_name,
                    "image_id": img_id,
                    "center": center
                })


    with open(os.path.join("train_log", experiment_name, f'scoremap_centers_{threshold}.json') , 'w') as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
