import os
from .lt_data import LT_Dataset


class Food101_LT(LT_Dataset):
    classnames_txt = "./datasets/Food101_LT/classnames.txt"
    train_txt = "./datasets/Food101_LT/Food101_train_LT.txt"
    test_txt = "./datasets/Food101_LT/Food101_test_LT.txt"

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
                # folder = line[0]
                classname = " ".join(line[1:])
                classnames.append(classname)
        return classnames
    