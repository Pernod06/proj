import cv2
import numpy as np


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts = param['pts']
        img = param['img']
        if len(pts) < 4:
            pts.append([x, y])  # 只记录前四次鼠标左击的位置
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)  # 画红点
            cv2.imshow('img1', img)
        else:
            print("已收集到四个点，准备进行透视变换。")
            cv2.destroyWindow('img1')  # 第五次鼠标左击直接关闭图片窗口


def Pic_correct(img):
    # img = cv2.imread(image_path)
    # if img is None:
    #     print(f"无法读取图像 {image_path}")
    #     return
    pts = []
    param = {'pts': pts, 'img': img}

    cv2.namedWindow('img1')
    cv2.setMouseCallback('img1', click, param)
    cv2.imshow('img1', img)
    cv2.waitKey(0)

    # 确定点是否按顺序排列，这里假设用户按顺时针顺序点击
    def sort_points(points):
        points.sort(key=lambda p: (p[1], p[0]))
        top_left, bottom_left, top_right, bottom_right = sorted(points, key=lambda p: p[0] + p[1])
        return [top_left, top_right, bottom_right, bottom_left]

    pts = sort_points(pts)
    print("Selected points:", pts)

    width, height = 250, 100
    pts1 = np.float32(pts)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img2 = cv2.warpPerspective(img, matrix, (width, height))

    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img2

# 使用方法
# Pic_correct('imgs/2/img_4.png')
# row_img = cv2.imread("imgs/img.png")
# pil_image = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
# # 定义蓝色范围（可能需要根据具体情况进行调整）
# lower_blue = np.array([100, 150, 0])  # 调整这些值以匹配您的蓝色
# upper_blue = np.array([140, 255, 240])
#
#     # 将图像转换为OpenCV格式
# open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
#
#     # 创建只包含蓝色的掩膜
# mask = cv2.inRange(open_cv_image, lower_blue, upper_blue)
# inverted_blue_mask = cv2.bitwise_not(mask)
# cv2.imshow('mask', mask)
# cv2.waitKey(0)
#
#     # 对掩膜应用形态学操作以去除噪音
# kernel = np.ones((2, 2), np.uint8)
# mask_cleaned = cv2.morphologyEx(inverted_blue_mask, cv2.MORPH_OPEN, kernel)  # 增加迭代次数
#
#
# mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
#
#     # 应用掩膜到原图上，得到只有蓝色背景和白色字体的图像
#
# blue_background = cv2.bitwise_and(row_img, row_img, mask=mask_cleaned)
#
#     # 将图像转换为灰度图像
# gray = cv2.cvtColor(blue_background, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)
#     # 进行二值化处理
# _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)
