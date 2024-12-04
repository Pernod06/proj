
import argparse
import copy
import os
from torchvision import transforms
import torch.nn.functional as F
from function import *
from model import *
from test import *
from ultralytics.nn.tasks import  attempt_load_weights
from plate_recognition.plate_rec import get_plate_result,init_model,cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

label_to_category = {
    'None': 'None',
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '10': 'A',
    '11': 'B',
    '12': 'C',
    '13': '川',
    '14': 'D',
    '15': 'E',
    '16': '鄂',
    '17': 'F',
    '18': 'G',
    '19': '赣',
    '20': '甘',
    '21': '贵',
    '22': '桂',
    '23': 'H',
    '24': '黑',
    '25': '沪',
    '26': 'J',
    '27': '冀',
    '28': '津',
    '29': '京',
    '30': '吉',
    '31': 'K',
    '32': 'L',
    '33': '辽',
    '34': '鲁',
    '35': 'M',
    '36': '蒙',
    '37': '闽',
    '38': 'N',
    '39': '宁',
    '40': 'P',
    '41': 'Q',
    '42': '青',
    '43': '琼',
    '44': 'R',
    '45': 'S',
    '46': '陕',
    '47': '苏',
    '48': '晋',
    '49': 'T',
    '50': 'U',
    '51': 'V',
    '52': 'W',
    '53': '皖',
    '54': 'X',
    '55': '湘',
    '56': '新',
    '57': 'Y',
    '58': '豫',
    '59': '渝',
    '60': '粤',
    '61': '云',
    '62': 'Z',
    '63': '藏',
    '64': '浙'
}

def add_black_borders(img):
    width, height = img.size
    new_width = 2 * width
    new_image = Image.new('RGB', (new_width, height), 'black')
    width = int(width / 2)
    new_image.paste(img, (width, 0))

    return new_image

def judge(in_put):
    top_values, top_indices = torch.topk(in_put, 3)
    v1 = top_values[0][0].item()
    v2 = top_values[0][1].item()
    v3 = top_values[0][2].item()
    label = top_indices[0][0].item()
    thresh_v = (v1 - v2 - v3) / v1
    if thresh_v > 0.5:
        return v1, label
    else:
        return None, None


def allFilePath(rootPath, allFIleList):  # 读取文件夹内的文件，放到list
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


def four_point_transform(image, pts):  # 透视变换得到车牌小图
    # rect = order_points(pts)
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def letter_box(img, size=(640, 640)):  # yolo 前处理 letter_box操作
    h, w, _ = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    new_img = cv2.resize(img, (new_w, new_h))
    left = int((size[1] - new_w) / 2)
    top = int((size[0] - new_h) / 2)
    right = size[1] - left - new_w
    bottom = size[0] - top - new_h
    img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, left, top


def load_model(weights, device):  # 加载yolov8 模型
    model = attempt_load_weights(weights, device=device)  # load FP32 model
    return model


def xywh2xyxy(det):  # xywh转化为xyxy
    y = det.clone()
    y[:, 0] = det[:, 0] - det[0:, 2] / 2
    y[:, 1] = det[:, 1] - det[0:, 3] / 2
    y[:, 2] = det[:, 0] + det[0:, 2] / 2
    y[:, 3] = det[:, 1] + det[0:, 3] / 2
    return y


def my_nums(dets, iou_thresh):  # nms操作
    y = dets.clone()
    y_box_score = y[:, :5]
    index = torch.argsort(y_box_score[:, -1], descending=True)
    keep = []
    while index.size()[0] > 0:
        i = index[0].item()
        keep.append(i)
        x1 = torch.maximum(y_box_score[i, 0], y_box_score[index[1:], 0])
        y1 = torch.maximum(y_box_score[i, 1], y_box_score[index[1:], 1])
        x2 = torch.minimum(y_box_score[i, 2], y_box_score[index[1:], 2])
        y2 = torch.minimum(y_box_score[i, 3], y_box_score[index[1:], 3])
        zero_ = torch.tensor(0).to(device)
        w = torch.maximum(zero_, x2 - x1)
        h = torch.maximum(zero_, y2 - y1)
        inter_area = w * h
        nuion_area1 = (y_box_score[i, 2] - y_box_score[i, 0]) * (y_box_score[i, 3] - y_box_score[i, 1])  # 计算交集
        union_area2 = (y_box_score[index[1:], 2] - y_box_score[index[1:], 0]) * (
                    y_box_score[index[1:], 3] - y_box_score[index[1:], 1])  # 计算并集

        iou = inter_area / (nuion_area1 + union_area2 - inter_area)  # 计算iou

        idx = torch.where(iou <= iou_thresh)[0]  # 保留iou小于iou_thresh的
        index = index[idx + 1]
    return keep


def restore_box(dets, r, left, top):  # 坐标还原到原图上

    dets[:, [0, 2]] = dets[:, [0, 2]] - left
    dets[:, [1, 3]] = dets[:, [1, 3]] - top
    dets[:, :4] /= r
    # dets[:,5:13]/=r

    return dets
    # pass


def post_processing(prediction, conf, iou_thresh, r, left, top):  # 后处理

    prediction = prediction.permute(0, 2, 1).squeeze(0)
    xc = prediction[:, 4:6].amax(1) > conf  # 过滤掉小于conf的框
    x = prediction[xc]
    if not len(x):
        return []
    boxes = x[:, :4]  # 框
    boxes = xywh2xyxy(boxes)  # 中心点 宽高 变为 左上 右下两个点
    score, index = torch.max(x[:, 4:6], dim=-1, keepdim=True)  # 找出得分和所属类别
    x = torch.cat((boxes, score, x[:, 6:14], index), dim=1)  # 重新组合

    score = x[:, 4]
    keep = my_nums(x, iou_thresh)
    x = x[keep]
    x = restore_box(x, r, left, top)
    return x


def pre_processing(img, opt, device):  # 前处理
    img, r, left, top = letter_box(img, (opt.img_size, opt.img_size))
    # print(img.shape)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()  # bgr2rgb hwc2chw
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img = img / 255.0
    img = img.unsqueeze(0)
    return img, r, left, top


def det_rec_plate(img, img_ori, detect_model, plate_rec_model):
    result_list = []
    img, r, left, top = pre_processing(img, opt, device)  # 前处理
    predict = detect_model(img)[0]
    # print(predict)
    outputs = post_processing(predict, 0.3, 0.5, r, left, top)  # 后处理
    for output in outputs:
        result_dict = {}
        output = output.squeeze().cpu().numpy().tolist()
        rect = output[:4]
        rect = [int(x) for x in rect]
        label = output[-1]
        roi_img = img_ori[rect[1]:rect[3], rect[0]:rect[2]]
        # land_marks=np.array(output[5:13],dtype='int64').reshape(4,2)
        # roi_img = four_point_transform(img_ori,land_marks)   #透视变换得到车牌小图
        if int(label):  # 判断是否是双层车牌，是双牌的话进行分割后然后拼接
            roi_img = get_split_merge(roi_img)
        plate_number, rec_prob, plate_color, color_conf = get_plate_result(roi_img, device, plate_rec_model,
                                                                           is_color=True)

        result_dict['plate_no'] = plate_number  # 车牌号
        result_dict['plate_color'] = plate_color  # 车牌颜色
        result_dict['rect'] = rect  # 车牌roi区域
        result_dict['detect_conf'] = output[4]  # 检测区域得分
        # result_dict['landmarks']=land_marks.tolist() #车牌角点坐标
        # result_dict['rec_conf']=rec_prob   #每个字符的概率
        result_dict['roi_height'] = roi_img.shape[0]  # 车牌高度
        # result_dict['plate_color']=plate_color
        # if is_color:
        result_dict['color_conf'] = color_conf  # 颜色得分
        result_dict['plate_type'] = int(label)  # 单双层 0单层 1双层
        result_list.append(result_dict)
    return result_list

def main():
    folder_path = r"imgs/2"

    model = CNN()
    model.load_state_dict(torch.load('new_model.pt'))
    power_list = []
    for filename in os.listdir(folder_path):

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)

            # 读取图片
            img = cv2.imread(file_path)
            img_ori = copy.deepcopy(img)
            list = det_rec_plate(img, img_ori, detect_model, plate_rec_model)
            print("读取的图片为：", list)
            rect = list[0]['rect']
            img = img_ori[rect[1]:rect[3], rect[0]:rect[2]]
            print("读取图片的形状为：", img.shape)
            img = Pic_correct(img)
            # 在这里对图片进行处理
            # 例如显示图片
            # cv2.imshow('Image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print(f'Processed {file_path}')
            pre_list = []
            seg_list = segment(img)
            print("seg_list长度为:", len(seg_list))
            cnt = 0
            for _img in seg_list:
                _img = Image.fromarray(_img)
                # _img = add_black_borders(_img)  # 使图像居中
                # _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
                # _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2BGR)

                _img = transform(_img)
                output = model(_img)
                output = F.softmax(output, dim=1)
                value, label = judge(output)

                # 更新power_list
                if cnt < len(power_list) and power_list[cnt][0] is not None:
                    if value is not None:
                        # if power_list[cnt][1] < value:
                        #     power_list[cnt][1] = value
                        # label 相同情况
                        if power_list[cnt][0] == label:
                            # 小于权值不更新，大于全值增强
                            if power_list[cnt][1] < value:
                                power_list[cnt][1] = value
                                power_list[cnt][2] += 2 % 10
                            else:
                                power_list[cnt][2] += 1 % 10
                        #label 不同情况
                        if power_list[cnt][0] != label:
                            # 大于权值
                            if power_list[cnt][1] <= value:
                                power_list[cnt][2] -= 2     # 削弱
                            else:
                                power_list[cnt] -= 1
                            if power_list[cnt][2] <= 0:
                                power_list[cnt] = [label, value, 1]
                else:
                    if len(power_list) >= 7:
                        power_list[cnt] = [label, value, 1]
                    else:
                        power_list.append([label, value, 1])

                if label is None and cnt <= len(power_list):
                    label = power_list[cnt][0]

                else:
                    if power_list[cnt][1] > value:
                        label = power_list[cnt][0]

                # print(label, value)
                pre_list.append(label_to_category[str(label)])
                cnt += 1
                # _, predicted = torch.max(output.data, 1)
                # pre_list.append(label_to_category[str(predicted.item())])
            print(cnt)
            while cnt < 7:
                pre_list.append(label_to_category[str(power_list[cnt][0])])
                cnt += 1
            print(pre_list)
            print(power_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default=r'weights/yolov8s.pt',
                        help='model.pt path(s)')  # yolov8检测模型
    parser.add_argument('--rec_model', type=str, default=r'weights/plate_rec_color.pth',
                        help='model.pt path(s)')  # 车牌字符识别模型
    parser.add_argument('--image_path', type=str, default=r'imgs/2', help='source')  # 待识别图片路径
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')  # yolov8 网络模型输入大小
    parser.add_argument('--output', type=str, default='result', help='source')  # 结果保存的文件夹
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detect_model = load_model(opt.detect_model, device)  # 初始化yolov8识别模型
    plate_rec_model = init_model(device, opt.rec_model, is_color=True)

    main()