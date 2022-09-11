import argparse
import json
import os
import pickle
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="./dataset/COCO/val")
    parser.add_argument('--experiment_name',
                        default="COCO_efficientnet_grad_cam")
    parser.add_argument(
        '--ann_file', default="./dataset/COCO/annotations/instances_val2014.json")
    parser.add_argument(
        '--category_file', default='./dataset/COCO/annotations/category.json', help='The category file ')
    parser.add_argument('--image-size', '-i', default=448, type=int,
                        metavar='N', help='image size (default: 448)')
    return parser.parse_args()


def bbox_x(cell):
    return cell[0]


def bbox_y(cell):
    return cell[1]


def bbox_w(cell):
    return cell[2]


def bbox_h(cell):
    return cell[3]


def center_x(cell):
    return cell[0]


def center_y(cell):
    return cell[1]


def distance_from_bbox(row):
    bbox_x, bbox_y, w, h = row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']

    # center of the bbox
    bbox_center_x = bbox_x + (w / 2.0)
    bbox_center_y = bbox_y + (h / 2.0)

    x, y = row['center_x'], row['center_y']

    # if the center is inside a bbox, then this metric is the euclidean distance
    # between the prediction point and the bbox center, else +inf
    if bbox_x <= x <= bbox_x + w and bbox_y <= y <= bbox_y + h:
        return np.sqrt((x - bbox_center_x) ** 2 + (y - bbox_center_y) ** 2)
    else:
        return np.inf


def mean_average_precision(predictions, coco, categories, image_size, cob_threshold=0):
    orig_cat_ids_to_names = {item['id']: item['name']
                             for _, item in coco.cats.items()}
    num_classes = len(categories)

    def resize_bbox(row):
        new_w, new_h = 448, 448
        img = coco.loadImgs(ids=row['image_id'])[0]
        orig_h, orig_w = img['height'], img['width']

        x, y, w, h = row['bbox']
        x = x * new_w / orig_w
        y = y * new_h / orig_h
        w = w * new_w / orig_w
        h = h * new_h / orig_h
        return [x, y, w, h]

    def old_id_to_new_id(old_id):
        name = orig_cat_ids_to_names[old_id]
        return categories[name]

    ground_truths = [gt for _, gt in coco.anns.items()]
    preds_df = pd.DataFrame(predictions)
    preds_df = preds_df.reset_index().rename(columns={"index": "p_id"})

    preds_df['center_x'] = preds_df['center'].apply(center_x)
    preds_df['center_y'] = preds_df['center'].apply(center_y)
    preds_df.drop('center', axis=1, inplace=True)

    gt_df = pd.DataFrame(ground_truths)
    gt_df = gt_df[gt_df['iscrowd'] == 0]
    gt_df['category_id'] = gt_df['category_id'].apply(old_id_to_new_id)  #
    gt_df.drop(['segmentation', 'area', 'iscrowd'], axis=1, inplace=True)
    gt_df['resized_bbox'] = gt_df.apply(resize_bbox, axis=1)
    gt_df['bbox_x'] = gt_df['resized_bbox'].apply(bbox_x)
    gt_df['bbox_y'] = gt_df['resized_bbox'].apply(bbox_y)
    gt_df['bbox_w'] = gt_df['resized_bbox'].apply(bbox_w)
    gt_df['bbox_h'] = gt_df['resized_bbox'].apply(bbox_h)
    gt_df.drop(['bbox', 'resized_bbox'], axis=1, inplace=True)
    gt_df.rename(columns={"id": "gt_id"}, inplace=True)

    # Merge predictions and ground truths df to calculate pair-wise distance
    # between predictions and bboxes of the same class and image
    merged_df = pd.merge(preds_df, gt_df, how="inner", left_on=['image_id', 'category_id'],
                         right_on=['image_id', 'category_id'])

    merged_df['distance_from_bbox'] = merged_df.apply(
        distance_from_bbox, axis=1)
    merged_df = merged_df.sort_values(
        ['category_id', 'image_id', 'distance_from_bbox'], ascending=True)

    CP = []
    CR = []
    APs = []
    epsilon = 1e-6

    for c in tqdm(range(num_classes)):
        df = merged_df[merged_df['category_id'] == c]

        visited_preds = set()
        visitied_gts = set()

        num_class_preds = len(df["p_id"].unique())
        num_ground_truths = len(df["gt_id"].unique())

        TP = np.zeros(num_class_preds)
        FP = np.zeros(num_class_preds)
        count = 0
        
        for _, row in df.iterrows():
            pred_id = row["p_id"]
            gt_id = row["gt_id"]

            # if this prediction is already checked, ignore
            if pred_id in visited_preds:
                continue

            # if this prediction is within a bbox, check if the bbox was not already "claimed" by another prediction
            if row["distance_from_bbox"] < np.inf:
                if gt_id not in visitied_gts:  # case 1
                    TP[count] = 1
                    visited_preds.add(pred_id)
                    visitied_gts.add(gt_id)
                    count += 1
                else:  # case 2
                    pass  # if gt_id is visited, nothing to conclude here as there maybe more bboxes for this prediction
                    # but there also maybe no more bboxes to match with this prediction, however we don't know
                    # about that at this point
            else:  # case 3
                FP[count] = 1
                visited_preds.add(pred_id)
                count += 1

        # Loop one more time to catch predictions missed from case 2 above
        for _, row in df.iterrows():
            pred_id = row["p_id"]
            if pred_id not in visited_preds:
                FP[count] = 1
                visited_preds.add(pred_id)
                count += 1

        TP_cumsum = np.cumsum(TP, axis=0)
        FP_cumsum = np.cumsum(FP, axis=0)

        precisions = np.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = np.concatenate((np.array([1]), precisions))

        recalls = TP_cumsum / (num_ground_truths + epsilon)
        recalls = np.concatenate((np.array([0]), recalls))

        APs.append(np.trapz(precisions, recalls))
        CP.append(precisions.tolist())
        CR.append(recalls.tolist())

    return CP, CR, APs, sum(APs) / len(APs)


def score():
    args = get_args()
    
    with open(args.ann_file, 'r') as f:
        coco = COCO(args.ann_file)

    with open(args.category_file, 'r') as f:
        categories = json.load(f)

    for threshold in [30, 50, 70]:
        with open(os.path.join("train_log", args.experiment_name, f"scoremap_centers_{threshold}.json"), 'r') as f:
            predictions = json.load(f)

        CP, CR, APs, mAP = mean_average_precision(
            predictions, coco, categories, args.image_size)

        output_file = os.path.join(
            "train_log", args.experiment_name, f"scoremap_centers_output_{threshold}.pickle")

        with open(output_file, 'wb') as f:
            pickle.dump({
                "CP": CP,
                "CR": CR,
                "APs": APs,
                "mAP": mAP
            }, f)

        print(f"mAP@{threshold}: {mAP}")


if __name__ == '__main__':
    score()
