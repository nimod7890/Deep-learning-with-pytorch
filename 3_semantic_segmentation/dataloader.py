import os.path as osp
from PIL import Image

import torch.utils.data as data
from utils.data_augumentation import (
    Compose,
    Scale,
    RandomRotation,
    RandomMirror,
    Resize,
    Normalize_Tensor,
)


class makeDatapathList:
    def __init__(self, rootpath):
        self.rootpath = rootpath
        self.img_path_template = osp.join(rootpath, "JPEGImage", "%s.jpg")
        self.anno_path_template = osp.join(rootpath, "SegmentationClass", "%s.png")

    def make_list(self, phase):
        id_names = osp.join(self.rootpath + f"ImageSets/Segmentation/{phase}.txt")
        img_list = []
        anno_list = []
        for line in open(id_names):
            file_id = line.strip()
            img_list.append(self.img_path_template % file_id)
            anno_list.append(self.anno_path_template % file_id)
        return [img_list, anno_list]

    def __call__(self, phase):
        return self.make_list(phase)


class dataTransform:
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train": Compose(
                [
                    Scale(scale=[0.5, 1.5]),
                    RandomRotation(angle=[-10, 10]),
                    RandomMirror(),
                    Resize(input_size),
                    Normalize_Tensor(color_mean, color_std),
                ]
            ),
            "val": Compose(
                [Resize(input_size), Normalize_Tensor(color_mean, color_std)]
            ),
        }

    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)


class VOCDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_file_path = self.img_list[index]
        img = Image.open(img_file_path)
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)
        return img, anno_class_img
