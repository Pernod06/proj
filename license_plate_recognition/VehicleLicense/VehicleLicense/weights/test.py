import torch
import torch.nn as nn
print(torch.__version__)
if torch.cuda.is_available():
    print("yes")
# content = torch.load('plate_rec_color.pth')
# print(content.keys())
# print(content['model'])