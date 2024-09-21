from __future__ import print_function, division
import re
import os
import math
import torch
import random
import warnings
import numpy as np
from PIL import Image
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")


def extract_and_combine_numbers(name):

    numbers = re.findall(r'\d+', name)

    combined_numbers = ''.join(numbers)

    return combined_numbers


class AUGAN_scene(Dataset):

    def __init__(self, maps=np.zeros(1), phase='train',
                 num1=0, num2=0,
                 data="data/",
                 numTx=80,
                 sample_size=1,
                 add_noise=False,
                 mean=0, sigma=30,                    # Noise mean and standard deviation initialization
                 sample_num=30,
                 transform=transforms.ToTensor()):

        if maps.size == 1:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps)
        else:
            self.maps = maps

        print('当前输入场景：AUGAN_scene3')
        self.data = data
        self.numTx = numTx
        self.transform = transform
        self.height = 256
        self.width = 256
        self.mean = mean
        self.sigma = sigma
        self.add_noise = add_noise
        self.sample_size = sample_size
        self.sample_num = sample_num
        self.num1 = num1
        self.num2 = num2

        if phase == 'train':
            self.num1 = 0
            self.num2 = 500
        elif phase == 'val':
            self.num1 = 501
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 601
            self.num2 = 700

        self.simulation = self.data+"image/"
        self.build = self.data + "build/"
        self.antenna = self.data + "antenna/"

    def __len__(self):
        return (self.num2-self.num1)*self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx/self.numTx).astype(int)
        idxc = idx-idxr*self.numTx
        dataset_map = self.maps[idxr+self.num1]

        name1 = str(dataset_map) + ".png"
        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        # loading target
        target_path = os.path.join(self.simulation, name2)
        target_image = Image.open(target_path)
        target_arr = np.asarray(target_image)

        # sampling
        numbers_combine = extract_and_combine_numbers(name2)  

        x_seed = '1' + numbers_combine  
        y_seed = '2' + numbers_combine  

        # Build whiteboard diagram (to store sampling points)
        sample_image = Image.new("L", target_image.size, "black")  
        num = 0
        # sample
        for i in range((self.width - self.sample_size) * (self.height - self.sample_size)):  

            # Generate random points along the upper left corner
            random.seed(int(x_seed + str(i)))
            x = random.randint(0, self.width - self.sample_size)  
            random.seed(int(y_seed + str(i)))
            y = random.randint(0, self.height - self.sample_size)

            # length * width  
            block = target_image.crop((x, y, x + self.sample_size, y + self.sample_size))

            if not np.any(np.any(np.array(block) == 0, axis=0)): 
                if self.add_noise:
                    arr_block = np.asarray(block)
                    # noise
                    gaussian_noise = np.random.normal(self.mean, self.sigma, 1)
                    # fuse
                    add_noise_block = arr_block + gaussian_noise
                    block = Image.fromarray(add_noise_block.astype(np.uint8))
                sample_image.paste(block, (x, y))  
                num = num + 1

            if num == self.sample_num:
                break

        # Does not contain a sample of the building
        sample_arr = np.asarray(sample_image)

        # building image
        build_arr = np.where(target_arr == 0, 255, 0) 

        # Contains a sampling of the building
        image_arr = sample_arr + build_arr

        # out_img = Image.fromarray(image_arr.astype('uint8'))
        # out_img.show()

        # generate masks
        mask_arr = np.where(image_arr == 0, 255, 0)  

        # transfer tensor
        arr_image = self.transform(image_arr / 255).type(torch.float32)
        arr_target = self.transform(target_arr).type(torch.float32)
        arr_mask = self.transform(mask_arr/255).type(torch.float32)

        return arr_image, arr_mask, arr_target, name2

