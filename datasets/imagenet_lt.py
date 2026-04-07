import os
from .lt_data import LT_Dataset
from torch.utils.data import Dataset
from PIL import Image


class ImageNet_LT(LT_Dataset):
    classnames_txt = "./datasets/ImageNet_LT/classnames.txt"
    train_txt = "./datasets/ImageNet_LT/ImageNet_LT_train.txt"
    test_txt = "./datasets/ImageNet_LT/ImageNet_LT_test.txt"

    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train, transform)

        self.classnames = self.read_classnames()

        self.names = []
        with open(self.txt) as f:
            for line in f:
                self.names.append(self.classnames[int(line.split()[1])])

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        name = self.names[index]
        return image, label, name

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames.append(classname)
        return classnames



class Sketch(Dataset):

    def __init__(self, root='/data00/jiahao/data/sketch', transform=None):
        # super().__init__(root, transform)


        self.images, self.labels = [], []
        self.transform=transform
        self.pre_process(root)

    def __getitem__(self, index):
        path = self.images[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        return len(self.images)

    def pre_process(self, root):
        self.name2id = {}
        file = '/data00/jiahao/PEL-main/datasets/ImageNet_LT/sketch_map.txt'
        with open(file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                name, idx = line.split(' ')
                self.name2id[name] = int(idx)

        files = os.listdir(root)
        for file in files:
            name_idx = self.name2id[file]
            path = os.path.join(root, file)
            images = os.listdir(path)
            for image in images:
                p = os.path.join(path, image)
                self.images.append(p)
                self.labels.append(name_idx)



class Sketch(Dataset):

    def __init__(self, root='/data00/jiahao/data/sketch', transform=None):
        # super().__init__(root, transform)


        self.images, self.labels = [], []
        self.transform=transform
        self.pre_process(root)

    def __getitem__(self, index):
        path = self.images[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        return len(self.images)

    def pre_process(self, root):
        self.name2id = {}
        file = '/data00/jiahao/PEL-main/datasets/ImageNet_LT/sketch_map.txt'
        with open(file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                name, idx = line.split(' ')
                self.name2id[name] = int(idx)

        files = os.listdir(root)
        for file in files:
            if 'README' in file or 'imagenet' in file:
                continue
            
            # name_idx = self.name2id[file]
            name_idx = int(file)
            path = os.path.join(root, file)
            images = os.listdir(path)
            for image in images:
                p = os.path.join(path, image)
                self.images.append(p)
                self.labels.append(name_idx)

