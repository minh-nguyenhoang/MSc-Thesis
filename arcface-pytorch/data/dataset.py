import os
import random
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys


class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(3, 112, 112)):
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

        # normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                # T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                # T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path)
        data = data.convert('L')
        data = self.transforms(data)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)



class ConfoundingDataset(data.Dataset):
    def __init__(self, data_list_file, phase='train', input_shape=(3, 112, 112)):
        self.phase = phase
        self.input_shape = input_shape

    
        self.data = self.process_data(data_list_file)

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

        # normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                # T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                # T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        ## Ensure data balancing on main label
        sample: str = self.data[index]
        img_path = random.choice(sample['image_files'])
        data = Image.open(img_path).convert('RGB').resize((112,112))
        data: torch.Tensor = self.transforms(data)
        label = torch.tensor(sample["label"]).long()
        gender_label = torch.tensor(sample['confounder_label']).long()
        return data.float(), label, gender_label

    def __len__(self):
        return len(self.data)
    
    def process_data(self, data_file_path):
        with open(data_file_path, 'r') as fd:
            lines = fd.readlines()

        result_lines = []
        for i, line in enumerate(lines):
            data_and_labels = line.split("\t")
            label = data_and_labels[0]
            confounder_label = data_and_labels[1]
            image_files = data_and_labels[2:]
            # image_files = [os.path.join(self.root, img) for img in image_files]
            result_lines.append({"label": i, 
                                        "label_name": label,
                                        "confounder_label": 0 if confounder_label == "M" else 1,
                                        "confounder_label_name": confounder_label, 
                                        "image_files": image_files})
        return result_lines


if __name__ == '__main__':
    dataset = Dataset(root='/data/Datasets/fv/dataset_v1.1/dataset_mix_aligned_v1.1',
                      data_list_file='/data/Datasets/fv/dataset_v1.1/mix_20w.txt',
                      phase='test',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)