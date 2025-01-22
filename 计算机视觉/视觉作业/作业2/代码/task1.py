import cv2
import numpy as np
from matplotlib import pyplot as plt


def harris_corner_detection(img_path):
    """
    Harris角点检测

    :param img_path: 输入图片的路径
    """
    # 读取图片
    image = cv2.imread(img_path)

    # 将图像转为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # 创建用于绘图的图形
    plt.figure(figsize=(15, 15), dpi=300)
    subplot_index = 1  # 记录当前子图的索引

    # 使用不同的参数进行Harris角点检测
    for k_size in range(1, 12, 2):
        for block_size in range(1, 6):
            # 计算Harris响应
            harris_response = cv2.cornerHarris(gray, block_size, k_size, 0.05)
            harris_response = cv2.dilate(harris_response, None)

            # 在图像上标记角点
            image[harris_response > 0.005 * harris_response.max()] = [255, 0, 0]

            # 绘制带有角点的图像
            plt.subplot(6, 5, subplot_index)
            subplot_index += 1
            plt.imshow(image, cmap='gray')
            plt.title(f'ksize: {k_size}, blockSize: {block_size}', fontsize=6), plt.axis('off')

    # 保存最终结果
    plt.savefig('harris_corner_detection_result.png')
    plt.show()


if __name__ == '__main__':
    harris_corner_detection('images/check_board.png')
