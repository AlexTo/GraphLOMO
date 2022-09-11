import torch.utils.data as data
import json
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from ._coco_utils import download_coco2014, gen_metadata


class COCO2014(data.Dataset):
    def __init__(self, data_dir="dataset/COCO",
                 metadata_dir="metadata/COCO",
                 transform=None,
                 phase='train',
                 emb_name=None):
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

        with open(emb_name, 'rb') as f:
            self.emb = pickle.load(f)
        self.emb_name = emb_name

    def get_anno(self):
        list_path = self.data_dir / f'{self.phase}_anno.json'
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(self.data_dir / 'category.json', 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        filename = item['file_name']
        labels = sorted(item['labels'])
        img_path = self.data_dir / f'{self.phase}2014' / filename
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, self.emb), target
