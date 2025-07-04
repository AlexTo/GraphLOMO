import numpy as np
import pickle
import os
import subprocess
import json
import torch.utils.data as data
from tqdm import tqdm
from collections import Counter
from pycocotools.coco import COCO

from PIL import Image
from pathlib import Path


urls = {
    "train_img": "http://images.cocodataset.org/zips/train2014.zip",
    "val_img": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}


def download_coco2014(data_dir, phase):
    os.makedirs(data_dir, exist_ok=True)
    if phase == "train":
        filename = "train2014.zip"
    else:
        filename = "val2014.zip"
    cached_file = data_dir / filename
    img_data = data_dir / filename.split(".")[0]

    if not os.path.exists(img_data):
        if not os.path.exists(cached_file):
            print(f"Downloading {urls[phase + '_img']}' to {cached_file}")
            subprocess.run(f"wget {urls[phase + '_img']} -P {data_dir}", shell=True)
        print(f"Extracting tar file {cached_file} to {data_dir}")
        subprocess.run(f"unzip {cached_file} -d {data_dir}", shell=True)

    # train/val images/annotations
    cached_file = data_dir / "annotations_trainval2014.zip"
    annotations_data = data_dir / "annotations"

    if not os.path.exists(annotations_data):
        if not os.path.exists(cached_file):
            print(f"Downloading {urls['annotations']} to {cached_file}")
            subprocess.run(f"wget {urls['annotations']} -P {data_dir}", shell=True)
        print(f"Extracting tar file {cached_file} to {data_dir}")
        subprocess.run(f"unzip {cached_file} -d {data_dir}", shell=True)

    anno = data_dir / f"{phase}_anno.json"
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):

        annotations_file = json.load(
            open(os.path.join(annotations_data, "instances_{}2014.json".format(phase)))
        )
        annotations = annotations_file["annotations"]
        category = annotations_file["categories"]
        category_id = {}
        for cat in category:
            category_id[cat["id"]] = cat["name"]
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file["images"]
        for annotation in annotations:
            if annotation["image_id"] not in annotations_id:
                annotations_id[annotation["image_id"]] = set()
            annotations_id[annotation["image_id"]].add(
                cat2idx[category_id[annotation["category_id"]]]
            )
        for img in images:
            if img["id"] not in annotations_id:
                continue
            if img["id"] not in img_id:
                img_id[img["id"]] = {}
            img_id[img["id"]]["file_name"] = img["file_name"]
            img_id[img["id"]]["labels"] = list(annotations_id[img["id"]])
        anno_list = []
        for _, v in img_id.items():
            anno_list.append(v)
        with open(anno, "w") as f:
            json.dump(anno_list, f)
        if not os.path.exists(data_dir / "category.json"):
            with open(data_dir / "category.json", "w") as f:
                json.dump(cat2idx, f)


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


def load_coco(data_dir):

    ann_file = data_dir / f"annotations/instances_train2014.json"
    coco = COCO(ann_file)

    cats = {k: v for k, v in sorted(coco.cats.items(), key=lambda cat: cat[1]["name"])}

    coco_id_to_new_id = {c: i for i, c in enumerate(cats)}
    return coco, coco_id_to_new_id


def adj_by_count(data_dir, metadata_dir):
    print("Generating adj matrix by co-occurrences")
    coco, coco_id_to_new_id = load_coco(data_dir)
    adj = np.zeros((80, 80)).astype(int)
    nums = np.zeros(80).astype(int)

    for idx in tqdm(coco.imgs):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=idx))
        labels = list(
            set([coco_id_to_new_id[a["category_id"]] for a in anns if not a["iscrowd"]])
        )
        n = len(labels)
        if n > 0:
            for i in range(n):
                nums[labels[i]] += 1
                for j in range(i + 1, n):
                    adj[labels[i]][labels[j]] += 1
                    adj[labels[j]][labels[i]] += 1

    result = {"nums": nums, "adj": adj}

    with open(metadata_dir / "topology" / f"coco_adj.pkl", "wb") as f:
        pickle.dump(result, f)


def adj_x_y(data_dir, metadata_dir, cof_x, cof_y):
    print(f"Generating {cof_x}_{cof_y} adj matrix")
    coco, coco_id_to_new_id = load_coco(data_dir)
    adj = np.zeros((80, 80))
    nums = np.zeros(80).astype(int)

    for idx in tqdm(coco.imgs):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=idx))
        labels = [coco_id_to_new_id[a["category_id"]] for a in anns if not a["iscrowd"]]
        label_count = Counter(labels)
        labels = list(label_count.keys())
        n = len(labels)
        if n > 0:
            for i in range(n):
                if label_count[labels[i]] >= cof_x:
                    nums[labels[i]] += 1
                for j in range(i + 1, n):
                    x = labels[i]
                    y = labels[j]
                    if label_count[x] >= cof_x and label_count[y] >= cof_y:
                        adj[x][y] += 1

                    if label_count[y] >= cof_x and label_count[x] >= cof_y:
                        adj[y][x] += 1

    result = {"nums": nums, "adj": adj}
    with open(metadata_dir / "topology" / f"coco_adj_{cof_x}_{cof_y}.pkl", "wb") as f:
        pickle.dump(result, f)


def adj_x_dot_y(data_dir, metadata_dir, cof_x, cof_y):
    print(f"Generating {cof_x}.{cof_y} adj matrix")
    coco, coco_id_to_new_id = load_coco(data_dir)
    adj = np.zeros((80, 80))
    nums = np.zeros(80).astype(int)

    for idx in tqdm(coco.imgs):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=idx))
        labels = [coco_id_to_new_id[a["category_id"]] for a in anns if not a["iscrowd"]]
        label_count = Counter(labels)
        labels = list(label_count.keys())
        n = len(labels)
        if n > 0:
            for i in range(n):
                if label_count[labels[i]] / cof_x >= 1:
                    nums[labels[i]] += 1
                for j in range(i + 1, n):
                    x = labels[i]
                    y = labels[j]
                    if label_count[x] / cof_x >= 1 and label_count[y] >= cof_y / cof_x:
                        adj[x][y] += 1

                    if label_count[y] / cof_y >= 1 and label_count[x] >= cof_x / cof_y:
                        adj[y][x] += 1

    result = {"nums": nums, "adj": adj}

    with open(metadata_dir / "topology" / f"coco_adj_{cof_x}_{cof_y}.pkl", "wb") as f:
        pickle.dump(result, f)


def gen_metadata(data_dir, metadata_dir):
    if not os.path.exists(metadata_dir / "topology"):
        os.makedirs(metadata_dir / "topology", exist_ok=True)
        adj_by_count(data_dir, metadata_dir)
        for x, y in zip([1, 1, 1, 1, 2, 3, 4], [1, 2, 3, 4, 1, 1, 1]):
            adj_x_y(data_dir, metadata_dir, x, y)

    phases = ["train", "val"]

    if not os.path.exists(metadata_dir / "train") and not os.path.exists(
        metadata_dir / "val"
    ):

        os.makedirs(metadata_dir / "train", exist_ok=True)
        os.makedirs(metadata_dir / "val", exist_ok=True)

        for phase in phases:
            ann_file = data_dir / "annotations" / f"instances_{phase}2014.json"
            coco = COCO(ann_file)
            cats = coco.cats
            cats = {
                k: v for k, v in sorted(cats.items(), key=lambda cat: cat[1]["name"])
            }

            coco_id_to_new_id = {c: i for i, c in enumerate(cats)}

            image_ids = []
            class_labels = []
            bboxes = []
            image_sizes = []

            for idx in coco.imgs:
                img = coco.imgs[idx]
                anns = coco.loadAnns(coco.getAnnIds(imgIds=idx))
                image_ids.append(img["file_name"])
                class_labels.append(
                    [
                        coco_id_to_new_id[a["category_id"]]
                        for a in anns
                        if not a["iscrowd"]
                    ]
                )
                bboxes.append(
                    [
                        [
                            a["bbox"][0],
                            a["bbox"][1],
                            a["bbox"][0] + a["bbox"][2],
                            a["bbox"][1] + a["bbox"][3],
                        ]
                        for a in anns
                        if not a["iscrowd"]
                    ]
                )  # convert to x0, y0, x1, y1 format
                image_sizes.append([img["width"], img["height"]])

            with open(
                metadata_dir / f"{phase}/image_ids.txt", "w"
            ) as image_ids_file, open(
                metadata_dir / f"{phase}/image_sizes.txt", "w"
            ) as image_sizes_file, open(
                metadata_dir / f"{phase}/class_labels.txt", "w"
            ) as class_labels_file, open(
                metadata_dir / f"{phase}/localization.txt", "w"
            ) as localization_file:

                for i in range(len(image_ids)):
                    img_id = image_ids[i]
                    if len(class_labels[i]) == 0:
                        continue
                    image_ids_file.write(f"{phase}2014/{img_id}\n")
                    class_labels_file.write(
                        f"{phase}2014/{img_id},{','.join(map(str, np.unique(class_labels[i])))}\n"
                    )
                    image_sizes_file.write(
                        f"{phase}2014/{img_id},{','.join(map(str,image_sizes[i]))}\n"
                    )

                    for j in range(len(bboxes[i])):
                        bbox = bboxes[i][j]
                        bbox_label = class_labels[i][j]
                        localization_file.write(
                            f"{phase}2014/{img_id},{bbox_label},{','.join(map(str, bbox))}\n"
                        )


class COCO2014(data.Dataset):
    def __init__(
        self,
        data_dir="data/coco",
        metadata_dir="metadata/coco",
        transform=None,
        phase="train",
        emb_name=None,
    ):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if isinstance(metadata_dir, str):
            metadata_dir = Path(metadata_dir)

        self.data_dir = data_dir
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(data_dir, phase)
        gen_metadata(data_dir, metadata_dir)
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        with open(emb_name, "rb") as f:
            self.emb = pickle.load(f)
        self.emb_name = emb_name

    def get_anno(self):
        list_path = self.data_dir / f"{self.phase}_anno.json"
        self.img_list = json.load(open(list_path, "r"))
        self.cat2idx = json.load(open(self.data_dir / "category.json", "r"))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        filename = item["file_name"]
        labels = sorted(item["labels"])
        img_path = self.data_dir / f"{self.phase}2014" / filename
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, self.emb), target
