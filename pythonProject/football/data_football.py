import cv2
import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

class FootballDataset(Dataset):
    def __int__(self, root):
        self.matches