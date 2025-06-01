import os
import random  # Rastgele örnekleme için
import numpy as np
from PIL import Image, ImageFile
import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F_func  # Alias for torch.nn.functional

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

##########################################
# Dataset: Domain A (RGB) & Domain B (S2)
##########################################
class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform_A=None, transform_B=None, sample_percentage=1.0):
        self.files_A = sorted(os.listdir(root_A))
        self.files_B = sorted(os.listdir(root_B))
        # Eğer sentinel2 dosyalarının sadece %10'unu kullanmak istiyorsanız:
        if sample_percentage < 1.0:
            sample_size = max(1, int(len(self.files_A) * sample_percentage))
            self.files_A = random.sample(self.files_A, sample_size)
        self.root_A = root_A
        self.root_B = root_B
        self.transform_A = transform_A
        self.transform_B = transform_B
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    def __getitem__(self, index):
        # Domain A: RGB image
        A_path = os.path.join(self.root_A, self.files_A[index % len(self.files_A)])
        img_A = Image.open(A_path).convert('RGB')
        if self.transform_A:
            img_A = self.transform_A(img_A)
        # Domain B: Sentinel-2 image with error handling
        attempts = 0
        while attempts < len(self.files_B):
            B_path = os.path.join(self.root_B, self.files_B[index % len(self.files_B)])
            try:
                img_B = tiff.imread(B_path)  # Expected shape: (channels, height, width)
                if img_B.ndim == 3 and img_B.shape[0] in [1, 13]:
                    img_B = np.moveaxis(img_B, 0, -1)  # Convert to (height, width, channels)
                if self.transform_B:
                    img_B = self.transform_B(img_B)
                break
            except Exception as e:
                print(f"Skipping file {B_path} due to error: {e}")
                index = (index + 1) % len(self.files_B)
                attempts += 1
        else:
            raise RuntimeError("No valid Sentinel-2 file found in dataset.")
        return {'A': img_A, 'B': img_B}

##############################################
# Custom Transforms for Sentinel-2 (Domain B) #
##############################################
class SentinelToTensor(object):
    def __call__(self, x):
        # x: numpy array with shape (H, W, C) where C is expected (e.g., 13)
        # Convert to torch tensor with shape (C, H, W)
        return torch.from_numpy(x).permute(2, 0, 1).float()

class SentinelResize(object):
    def __init__(self, size):
        self.size = size  # size: (height, width)
    def __call__(self, x):
        # x: torch tensor of shape (C, H, W)
        x = F_func.interpolate(x.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        return x

#######################################
# Channel Attention Module (SE Block) #
#######################################
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

#########################
# ResNet Block for Generator
#########################
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

########################################
# Generator with Channel Attention
# (For conditional generation, G outputs only extra channels (10))
########################################
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features * 2
        # Residual Blocks
        for _ in range(n_residual_blocks):
            model += [ResnetBlock(in_features)]
        # Channel Attention Module
        model += [ChannelAttention(in_features, reduction=16)]
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features // 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),  # output_nc will be 10 for G
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

#########################
# PatchGAN Discriminator#
#########################
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, True)
        ]
        model += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)