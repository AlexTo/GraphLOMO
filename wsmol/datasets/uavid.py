import os
import torch.utils.data as data
import json
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from .coco import download_coco2014, gen_metadata


def download_uavid(data_dir: Path, phase: str):
    if not os.path.exists(data_dir / "meta.json"):
        raise RuntimeError(
            f"{data_dir / "meta.json"} does not exist. Please download UAVid dataset at https://datasetninja.com/uavid#download"
        )


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


def gen_metadata(data_dir: Path, metadata_dir: Path):
    pass


class UAVid(data.Dataset):
    def __init__(
        self,
        data_dir="data/uavid",
        metadata_dir="metadata/uavid",
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
        download_uavid(data_dir, phase)
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
