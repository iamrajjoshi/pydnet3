import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from base_dataloader import BaseDataset, get_params, get_transform, normalize


class StereoDataloader_test(Dataset):
    __left = []

    def __init__(self, opt):
        self.opt = opt

        for root, dirs, files in os.walks(self.opt.dataroot):
            for file in files:
                if file.endswith(".jpg"):
                    file_path = os.path.join(root, file)
                    if "Camera_0" in file_path:
                        self.__left.append(file_path)

    def __getitem__(self, index):
        left_img = Image.open(self.__stereo[index]).convert('RGB')
        
        params = get_params(self.opt, left_img.size)

        transform = get_transform(self.opt, params)
        left_img = transform(left_img)

        input_dict = {'test_img': left_img.cuda()}

        return input_dict

    def __len__(self):
        return len(self.__stereo)
