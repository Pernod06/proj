import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
import torch.nn as nn
from datasets import tqdm
from torch import optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model import LeNet5



# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((20, 20)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, img_path, mode, transform=None):

        super(MyDataset, self).__init__()
        self.root = img_path
        self.txt_root = os.path.join(self.root, str(mode)+'.txt')
        with open(self.txt_root, 'r') as f:
            data = f.readlines()

        self.imgs = []
        self.labels = []
        for line in data:
            line = line.rstrip()
            word = line.split()
            img_path = os.path.join(self.root)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            self.imgs.append(os.path.join(self.root, word[0]))
            self.labels.append(int(word[1]))  # 确保标签是整数
        self.transform = transform


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img_path = self.imgs[item]
        label = self.labels[item]
        # 打印路径以调试
        # print(f"Opening image: {img_path}")
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)  # 确保标签是 long 类型
        return img, label

# 数据集路径
path = 'D:\Project\VehicleLicense\VehicleLicense'

train_dataset = MyDataset(path, 'train',transform=transform)
test_dataset = MyDataset(path, 'test',transform=transform)
# 创建 DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
# 显示数据集中的图片
# for i, data in enumerate(data_loader):
#     images, labels = data
#
#     img = torchvision.utils.make_grid(images).numpy()
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#     plt.show()
#     break

model = LeNet5()
# model = AlexNet()
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):

    loop = tqdm(enumerate(train_loader), total = len(train_loader))
    running_loss = 0.0
    right = 0
    for step, (images, labels) in loop:
        if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        right += (predicted == labels).sum()

        loop.set_description(f'Epoch [{epoch}/{20}]')
        loop.set_postfix(loss=running_loss/(step+1), acc=float(right)/float(64*step+len(images)))

torch.save(model.state_dict(), 'model.pt')

model.eval()
loop = tqdm(enumerate(test_loader), total = len(test_loader))
right = 0
for step, (images, labels) in loop:
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
output = model(images)
loss = criterion(output, labels)

_, predicted = torch.max(output.data, 1)
right += (predicted == labels).sum()

loop.set_postfix(acc=float(right)/float(64*step+len(images)))
print('Test Accuracy: {:.2f}%'.format(100 * right / len(images)))




