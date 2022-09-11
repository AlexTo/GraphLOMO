import os
import os.path
import pickle
import torch.utils.data as data

from PIL import Image
from pathlib import Path
from ._voc_utils import download_voc2007, gen_metadata, read_object_labels, write_object_labels_csv, object_categories, read_object_labels_csv



class Voc2007Classification(data.Dataset):
    def __init__(self, data_dir="dataset/VOC",
                 metadata_dir="metadata/VOC", phase="train",
                 transform=None,
                 target_transform=None,
                 emb_name=None):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        if isinstance(metadata_dir, str):
            metadata_dir = Path(metadata_dir)

        self.data_dir = data_dir
        self.metadata_dir = metadata_dir

        self.path_devkit = data_dir / 'VOCdevkit'
        self.path_images = data_dir / 'VOCdevkit' / 'VOC2007' / 'JPEGImages'
        self.set = phase
        self.transform = transform
        self.target_transform = target_transform

        # download dataset
        download_voc2007(self.data_dir)

        gen_metadata(self.data_dir, self.metadata_dir)

        # define path of csv file
        path_csv = self.data_dir / 'files' / 'VOC2007'
        # define filename of csv file
        file_csv = path_csv / f'classification_{phase}.csv'

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            os.makedirs(path_csv, exist_ok=True)
            # generate csv file
            labeled_data = read_object_labels(
                self.data_dir, 'VOC2007', self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        with open(emb_name, 'rb') as f:
            self.emb = pickle.load(f)
        self.emb_name = emb_name

        print(
            f'VOC 2007 classification set={phase} number of classes={len(self.classes)}  number of images=len(self.images)')

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images,
                                      path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path, self.emb), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
