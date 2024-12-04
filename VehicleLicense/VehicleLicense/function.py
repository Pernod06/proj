import cv2
import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 使用PIL库读取图像
pil_image = Image.open("Infer\\test19.png")
# 将图像转换为OpenCV格式
open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 将图像转换为灰度图像
gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)


# 进行二值化处理
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# 分割图像
def find_end(start, arg, black, white, width, black_max, white_max):
    end = start + 1
    for m in range(start + 1, width - 1):
        if (black[m] if arg else white[m]) > (0.95*black_max if arg else 0.95*white_max):
            end = m
            break
    return end


def char_segmentation(thresh):
    """ 分割字符 """
    white, black = [], []    # list记录每一列的黑/白色像素总和
    height, width = thresh.shape
    white_max = 0    # 仅保存每列，取列中白色最多的像素总数
    black_max = 0    # 仅保存每列，取列中黑色最多的像素总数
    # 计算每一列的黑白像素总和
    for i in range(width):
        line_white = 0    # 这一列白色总数
        line_black = 0    # 这一列黑色总数
        for j in range(height):
            if thresh[j][i] == 255:
                line_white += 1
            if thresh[j][i] == 0:
                line_black += 1
        white_max = max(white_max, line_white)
        black_max = max(black_max, line_black)
        white.append(line_white)
        black.append(line_black)
        # print('white_max', white_max)
        # print('black_max', black_max)
    # arg为true表示黑底白字，False为白底黑字
    arg = True
    if black_max < white_max:
        arg = False

    # 分割车牌字符
    n = 1
    while n < width - 2:
        n += 1
        # 判断是白底黑字还是黑底白字  0.05参数对应上面的0.95 可作调整
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):  # 这点没有理解透彻
            start = n
            end = find_end(start, arg, black, white, width, black_max, white_max)
            n = end
            if end - start > 20 or end > (width * 3 / 7):
                cropImg = thresh[0:height, start-1:end+1]
                # 对分割出的数字、字母进行resize并保存
                cropImg = cv.resize(cropImg, (34, 56))
                cv.imwrite(save_path + '\\{}.bmp'.format(n), cropImg)
                cv.imshow('Char_{}'.format(n), cropImg)

save_path = 'D:\pyrorch-project\license_plate_recognition\VehicleLicense\VehicleLicense\segmPict1'

char_segmentation(thresh)