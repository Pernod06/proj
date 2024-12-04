import cv2
import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 使用PIL库读取图像
# cv2.imshow("1", thresh)
# cv.waitKey(0)
# 分割图像
def find_end(start, arg, black, white, width, black_max, white_max):
    end = start + 1
    for m in range(start + 1, width - 1):
        if (black[m] if arg else white[m]) > (0.95*black_max if arg else 0.95*white_max):
            end = m
            break
    return end


def char_segmentation(thresh, raw_img):
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
    ret = []
    while n < width - 2:
        n += 1
        # 判断是白底黑字还是黑底白字  0.05参数对应上面的0.95 可作调整
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):  # 这点没有理解透彻
            start = n
            end = find_end(start, arg, black, white, width, black_max, white_max)
            n = end
            if end - start > 20 or end > (width * 3 / 7):
                cropImg = raw_img[0:height, start-1:end+1]
                # 对分割出的数字、字母进行resize并保存
                cropImg = cv.resize(raw_img, (34, 56))
                ret.append(cropImg)

    return ret

def segment(row_img):
    pil_image = row_img
    # 定义蓝色范围（可能需要根据具体情况进行调整）
    lower_blue = np.array([100, 150, 0])  # 调整这些值以匹配您的蓝色
    upper_blue = np.array([140, 255, 240])

    # 将图像转换为OpenCV格式
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # 创建只包含蓝色的掩膜
    mask = cv2.inRange(open_cv_image, lower_blue, upper_blue)
    mask = cv2.bitwise_not(mask)
    # 对掩膜应用形态学操作以去除噪音
    kernel = np.ones((1, 1), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 增加迭代次数

    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    # 应用掩膜到原图上，得到只有蓝色背景和白色字体的图像
    blue_background = cv2.bitwise_and(row_img, row_img, mask=mask_cleaned)

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(blue_background, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)
    # 进行二值化处理
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # row_start, row_end = preprocess(thresh)
    # print('row_start:', row_start, 'row_end: ', row_end)

    # img = row_img[row_start:row_end, :]
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(thresh)
    # plt.show()
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)

    seg_index = col_seg(thresh)
    print("seg_index为：", seg_index)

    seg_img = []
    for i in range(len(seg_index)):
        img_ = thresh[:, seg_index[i][0]:seg_index[i][1]]
        seg_img.append(img_)

    for i, img in enumerate(seg_img):
        output_path = f"segmPict1/{i}.png"
        cv2.imwrite(output_path, img)
        print(f"Image saved to {output_path}")
    return seg_img



def preprocess(img):  #去除上下噪音
    hight, width = img.shape
    times_row = []
    for row in range(hight):
        pc = 0
        for col in range(width):
            if col != width-1:
                if img[row][col+1] != img[row][col]:
                    pc = pc + 1
        times_row.append(pc)
    # print("每行跳变的次数:", times_row)
    row_end = 0
    row_start = 0
    for row in range(hight-2):
        if times_row[row] < 12:
            continue
        elif times_row[row+1] < 12:
            continue
        elif times_row[row+2] < 12:
            continue
        else:
            row_start = row - 3
            break
    for row in range(row_start+10, hight-2):
        if times_row[row] > 12:
            continue
        elif times_row[row+1] > 12:
            continue
        elif times_row[row+2] > 12:
            continue
        else:
            row_end = row + 2
            break
    return row_start, row_end

def col_seg(img):
    lst_heise = []   # 记录每一列中的白色像素点数量
    seg_index = []
    hight, width = img.shape
    for i in range(width):
        pc = 0
        for j in range(hight):
            if img[j][i] == 255:
                pc += 1
        lst_heise.append(pc)
    j = 0
    for i in range(len(lst_heise)):
        if (lst_heise[i] == 0):
            if i - j > 20:
                seg_index.append((j, i))
            j = i
    if len(seg_index) < 7:
        seg_index.append((j, width))
    return seg_index



if __name__ == "__main__":
    pil_image = Image.open("imgs/img_2.png")
    # 将图像转换为OpenCV格式
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img = cv2.imread("imgs/img_2.png")
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    # 进行二值化处理
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    row_start, row_end = preprocess(thresh)
    print(row_start, row_end)

    img = img[row_start:row_end, :]
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_, 100, 255, cv2.THRESH_BINARY)
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(thresh)
    # plt.show()

    seg_index = col_seg(thresh)
    print(seg_index)

    seg_img = []
    for i in range(len(seg_index)):
        img_ = img[:, seg_index[i][0]:seg_index[i][1]]
        seg_img.append(img_)

    for i, img in enumerate(seg_img):
        output_path = f"segmPict1/{i}.png"
        cv2.imwrite(output_path, img)
        print(f"Image saved to {output_path}")





