import glob
import os.path

import cv2 as cv
import numpy as np
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

if __name__ == '__main__':
    # 设置是否显示图像的标志
    show = False

    # 定义图像路径和真值路径
    img_path = 'data set/BSDS500/images/test'
    gt_path = 'data set/BSDS500/GT_convert_0/test'

    # 获取图像路径列表
    img_list = glob.glob(f'{img_path}/*jpg')

    # 初始化用于存储精度和召回率的列表
    precision_list = []
    recall_list = []

    # 遍历图像列表
    for img_pth in tqdm(img_list):
        # 提取图像文件名
        img_name = os.path.basename(img_pth)

        # 读取灰度图像
        img = cv.imread(img_pth, 0)

        # 读取对应的真值图像
        y_true = cv.imread(f'{gt_path}/{img_name}', 0)

        # 使用Canny边缘检测生成预测图像
        y_pred = cv.Canny(img, 180, 240)

        # 对真值图像进行二值化处理
        _, y_true = cv.threshold(y_true, 128, 255, cv.THRESH_BINARY)

        # 如果设置了显示标志，展示图像
        if show:
            cv.imshow('1', y_pred)
            cv.imshow('2', y_true)
            cv.waitKey(0)

        # 计算并存储精度和召回率
        precision = precision_score(y_true.astype(np.uint8), y_pred.astype(np.uint8), average='micro')
        recall = recall_score(y_true.astype(np.uint8), y_pred.astype(np.uint8), average='micro')

        precision_list.append(precision)
        recall_list.append(recall)

    # 打印平均精度和召回率
    print("precision", np.mean(precision_list))
    print('recall', np.mean(recall_list))
