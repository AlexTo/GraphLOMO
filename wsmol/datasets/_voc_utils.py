from urllib.parse import urlparse
import pandas as pd
import numpy as np
import pickle
import subprocess
import os
import csv
import torch
from .. import util
from tqdm import tqdm
from collections import Counter
from bs4 import BeautifulSoup


object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

cats_dict = {c: i for i, c in enumerate(object_categories)}


def load_voc(data_dir):
    return pd.read_csv(data_dir / "files" / "VOC2007" / "classification_train.csv")


object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(
            path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(
                    row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images


def download_voc2007(root):
    path_devkit = root / 'VOCdevkit'
    path_images = root / 'VOCdevkit' / 'VOC2007' / 'JPEGImages'

    # create directory

    os.makedirs(root, exist_ok=True)

    if not os.path.exists(path_devkit):
        parts = urlparse(urls['devkit'])
        filename = os.path.basename(parts.path)
        cached_file = root / filename

        if not os.path.exists(cached_file):
            print(f"Downloading {urls['devkit']} to {cached_file}")
            util.download_url(urls['devkit'], cached_file)

        # extract file
        print(f'Extracting tar file {cached_file} to {root}')
        subprocess.run(f"tar -xvf {cached_file} -C {root}", shell=True)

    # train/val images/annotations
    if not os.path.exists(path_images):

        # download train/val images/annotations
        parts = urlparse(urls['trainval_2007'])
        filename = os.path.basename(parts.path)
        cached_file = root / filename

        if not os.path.exists(cached_file):
            print(f"Downloading {urls['trainval_2007']} to {cached_file}")
            util.download_url(urls['trainval_2007'], cached_file)

        # extract file
        print(f'Extracting tar file {cached_file} to {root}')
        subprocess.run(f"tar -xvf {cached_file} -C {root}", shell=True)

    # test annotations
    test_anno = os.path.join(
        path_devkit, 'VOC2007/ImageSets/Main/aeroplane_test.txt')
    if not os.path.exists(test_anno):

        # download test annotations
        parts = urlparse(urls['test_images_2007'])
        filename = os.path.basename(parts.path)
        cached_file = root / filename

        if not os.path.exists(cached_file):
            print(f"Downloading: {urls['test_images_2007']} to {cached_file}")
            util.download_url(urls['test_images_2007'], cached_file)

        # extract file
        print(f'Extracting tar file {cached_file} to {root}')
        subprocess.run(f"tar -xvf {cached_file} -C {root}", shell=True)

    # test images
    test_image = os.path.join(path_devkit, 'VOC2007/JPEGImages/000001.jpg')
    if not os.path.exists(test_image):

        # download test images
        parts = urlparse(urls['test_anno_2007'])
        filename = os.path.basename(parts.path)
        cached_file = root / filename

        if not os.path.exists(cached_file):
            print(f"Downloading: {urls['test_anno_2007']} to {cached_file}")
            util.download_url(urls['test_anno_2007'], cached_file)

        # extract file
        print(f'Extracting tar file {cached_file} to {root}')
        subprocess.run(f"tar -xvf {cached_file} -C {root}",  shell=True)


def gen_metadata(data_dir, metadata_dir):
    if not os.path.exists(metadata_dir / "topology"):
        os.makedirs(metadata_dir / "topology", exist_ok=True)
        adj_by_count(data_dir, metadata_dir)
        for x, y in zip([1, 1, 1, 1, 2, 3, 4], [1, 2, 3, 4, 1, 1, 1]):
            adj_x_y(data_dir, metadata_dir, x, y)

    phases = ["train", "val"]

    if os.path.exists(metadata_dir / "train") and os.path.exists(metadata_dir / "val"):
        return

    os.makedirs(metadata_dir / "train", exist_ok=True)
    os.makedirs(metadata_dir / "val", exist_ok=True)

    object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']
    cats_dict = {c: i for i, c in enumerate(object_categories)}

    for phase in phases:

        df = pd.read_csv(data_dir / "files" / "VOC2007" / f"classification_{phase}.csv")
        file_list = [str(f).zfill(6) for f in df['name'].values]

        try:
            image_ids_file = open(metadata_dir / f"{phase}/image_ids.txt", 'w')
            class_labels_file = open(
                metadata_dir / f"{phase}/class_labels.txt", 'w')
            image_sizes_file = open(
                metadata_dir / f"{phase}/image_sizes.txt", 'w')
            localization_file = open(
                metadata_dir / f"{phase}/localization.txt", 'w')

            for i, f in tqdm(enumerate(file_list)):
                img_id = f"VOCdevkit/VOC2007/JPEGImages/{f}.jpg"
                image_ids_file.write(f"{img_id}\n")
                labels = np.where(df.iloc[i, 1:] >= 0)[0]
                class_labels_file.write(
                    f"{img_id},{','.join(map(str, labels))}\n")

                with open(data_dir / f"VOCdevkit/VOC2007/Annotations/{f}.xml") as f:
                    img_metadata = BeautifulSoup(f, features="lxml")

                image_sizes_file.write(
                    f"{img_id},{img_metadata.size.width.text},{img_metadata.size.height.text}\n")
                objects = img_metadata.find_all("object")
                for o in objects:
                    bbox_label = cats_dict[o.find("name").text]
                    localization_file.write(
                        f"{img_id},{bbox_label},{o.find('xmin').text},{o.find('ymin').text},{o.find('xmax').text},{o.find('ymax').text}\n")

        finally:
            image_ids_file.close()
            class_labels_file.close()
            image_sizes_file.close()
            localization_file.close()


def adj_by_count(data_dir, metadata_dir):
    print("Generating adj matrix by co-occurrences")
    classif_train = load_voc(data_dir)
    adj = np.zeros((20, 20)).astype(int)
    nums = np.zeros(20).astype(int)
    n = len(object_categories)
    for i, row in tqdm(classif_train.iterrows()):
        onehot = row.values[1:]
        for i in range(n):
            if onehot[i] >= 0:
                nums[i] += 1
                for j in range(i+1, n):
                    if onehot[j] >= 0:
                        adj[i][j] += 1
                        adj[j][i] += 1

    result = {'nums': nums, 'adj': adj}
    with open(metadata_dir / "topology" / f"voc_adj.pkl", 'wb') as f:
        pickle.dump(result, f)


def adj_x_y(data_dir, metadata_dir, cof_x, cof_y):
    print(f"Generating {cof_x}_{cof_y} adj matrix")
    classif_train = load_voc(data_dir)
    adj = np.zeros((20, 20)).astype(int)
    nums = np.zeros(20).astype(int)

    for i, row in tqdm(classif_train.iterrows()):
        img_name = f"{str(row.values[0]).zfill(6)}.xml"
        labels0 = [i for i, v in enumerate(row.values[1:]) if v >= 0]
        img_xml = BeautifulSoup(
            open(data_dir / "VOCdevkit" / "VOC2007" / "Annotations" / img_name).read(), features="lxml")
        objects = img_xml.findAll("object")

        labels = [cats_dict[o.find("name").text] for o in objects]

        assert(sorted(labels0) == sorted(set(labels)))

        label_count = Counter(labels)
        labels = list(label_count.keys())
        n = len(labels)
        if n > 0:
            for i in range(n):
                if label_count[labels[i]] >= cof_x:
                    nums[labels[i]] += 1
                for j in range(i+1, n):
                    x = labels[i]
                    y = labels[j]
                    if label_count[x] >= cof_x and label_count[y] >= cof_y:
                        adj[x][y] += 1

                    if label_count[y] >= cof_x and label_count[x] >= cof_y:
                        adj[y][x] += 1

    result = {'nums': nums, 'adj': adj}

    with open(metadata_dir / "topology" / f"voc_adj_{cof_x}_{cof_y}.pkl", 'wb') as f:
        pickle.dump(result, f)


def adj_x_dot_y(data_dir, metadata_dir, cof_x, cof_y):
    print(f"Generating {cof_x}.{cof_y} adj matrix")
    classif_train = load_voc(data_dir)
    adj = np.zeros((20, 20)).astype(int)
    nums = np.zeros(20).astype(int)

    for i, row in tqdm(classif_train.iterrows()):
        img_name = f"{str(row.values[0]).zfill(6)}.xml"
        img_xml = BeautifulSoup(
            open(data_dir / "VOCdevkit" / "VOC2007" / "Annotations", img_name).read(), features="lxml")
        objects = img_xml.findAll("object")

        labels = [cats_dict[o.find("name").text] for o in objects]
        label_count = Counter(labels)
        labels = list(label_count.keys())

        n = len(labels)
        if n > 0:
            for i in range(n):
                if label_count[labels[i]] / cof_x >= 1:
                    nums[labels[i]] += 1
                for j in range(i+1, n):
                    x = labels[i]
                    y = labels[j]
                    if label_count[x] * cof_x >= label_count[y] * cof_y:
                        adj[x][y] += 1

                    if label_count[y] * cof_x >= label_count[x] * cof_y:
                        adj[y][x] += 1

    result = {'nums': nums, 'adj': adj}
    with open(metadata_dir / "topology" / f"voc_adj_{cof_x}.{cof_y}.pkl", 'wb') as f:
        pickle.dump(result, f)
