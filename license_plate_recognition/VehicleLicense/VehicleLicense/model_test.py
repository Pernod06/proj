import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model import CNN

def add_black_borders(img):
    width, height = img.size
    new_width = 2 * width
    new_image = Image.new('RGB', (new_width, height), 'black')
    width = int(width / 2)
    new_image.paste(img, (width, 0))

    return new_image

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

model = CNN()
model.load_state_dict(torch.load('new_model.pt'))

path = 'segmPict1/0.png'
img = Image.open(path).convert('RGB')
print(img.size)
img = transform(img)
# img = add_black_borders(img)

print(img.size)
img_np = img.numpy().transpose((1, 2, 0))
plt.imshow(img_np)
plt.show()
print(img.size())
output = model(img)
print(output.size())
_, predicted = torch.max(output.data, 1)
print(predicted)