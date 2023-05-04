import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import random
import torchvision.transforms as transforms
from base_dataloader import BaseDataset, get_params, get_transform, normalize


# +
class StereoDataloader(Dataset):
    __left = []
    __right = []

    def __init__(self, opt):
        self.opt = opt
        
        for root, dirs, files in os.walk(self.opt.dataroot):
            for file in files:
                if file.endswith(".jpg"):
                    file_path = os.path.join(root, file)
                    if "Camera_0" in file_path:
                        self.__left.append(file_path)
                    else:
                        self.__right.append(file_path)

    def __getitem__(self, index):
        left_img = Image.open(self.__left[index]).convert('RGB')
        right_img = Image.open(self.__right[index]).convert('RGB')
        
        params = get_params(self.opt, left_img.size)

#         arg = random.random() > 0.5
#         if arg:
#             left_img, right_img = self.augument_image_pair(left_img, right_img)

        transform = get_transform(self.opt, params)

        left_img = transform(left_img)
        right_img = transform(right_img)
        
        input_dict = {'left_img': left_img.cuda(), 'right_img': right_img.cuda()}

        return input_dict

    def augument_image_pair(self, left_image, right_image):

        left_image = np.asarray(left_image)
        right_image = np.asarray(right_image)
        # print(np.amin(left_image))

        # randomly gamma shift
        random_gamma = random.uniform(0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = random.uniform(0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        # random_colors = [random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)]
        # white = np.ones((left_image.shape[0],left_image.shape[1]))
        # color_image = np.stack([white * random_colors[i] for i in range(3)], axis=2)
        # left_image_aug  *= color_image
        # right_image_aug *= color_image

        # saturate
        # left_image_aug  = np.clip(left_image_aug,  0, 1)
        # right_image_aug = np.clip(right_image_aug, 0, 1)

        left_image_aug = Image.fromarray(np.uint8(left_image_aug))
        right_image_aug = Image.fromarray(np.uint8(right_image_aug))

        return left_image_aug, right_image_aug

    def __len__(self):
        return len(self.__left)
