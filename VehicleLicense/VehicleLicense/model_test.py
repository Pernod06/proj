import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model import LeNet5

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

model = LeNet5()
model.load_state_dict(torch.load('model.pt'))

path = 'segmPict/313.bmp'
img = Image.open(path)
print(img.size)
img = transform(img)
print(img.size())
output = model(img)
print(output.size())
_, predicted = torch.max(output.data, 1)
print(predicted)