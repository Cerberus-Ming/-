import cv2 as cv
import os
from matplotlib import pyplot as plt


def apply_gaussian_laplacian(src_path, dst_path):

    original_image = cv.imread(src_path)

    # 尺度因子从1到11
    for sigma in range(1, 12):
        plt.figure(dpi=500, figsize=(15, 15))

        subplot_index = 1
        # 窗口大小从3*3到29*29，以2为间隔
        for kernel_size in range(3, 30, 2):
            # 高斯滤波
            blurred_image = cv.GaussianBlur(original_image, (kernel_size, kernel_size), sigma)
            blurred_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2RGB)

            # 将图像添加至子图
            plt.subplot(6, 6, subplot_index)
            plt.imshow(blurred_image)
            plt.title(f'Gaussian: {kernel_size}*{kernel_size}')
            plt.xticks([])
            plt.yticks([])
            subplot_index += 1

            # 拉普拉斯算子边缘检测
            laplacian_image = cv.Laplacian(blurred_image, -1, ksize=3)
            laplacian_image = cv.cvtColor(laplacian_image, cv.COLOR_BGR2RGB)

            # 将图像添加至子图
            plt.subplot(6, 6, subplot_index)
            plt.imshow(laplacian_image)
            plt.title(f'Laplacian: {kernel_size}*{kernel_size} sigma={sigma}')
            plt.xticks([])
            plt.yticks([])
            subplot_index += 1

        plt.suptitle(f'Sigma = {sigma}')

        if not os.path.isdir(dst_path):
            os.makedirs(dst_path)

        plt.savefig(f'./{dst_path}/sigma={sigma}.png')  # 导出为图片

    plt.close()


if __name__ == '__main__':
    grey_lena_path = 'data set/lena/lena512.bmp'
    color_lena_path = 'data set/lena/lena512color.tiff'
    
    apply_gaussian_laplacian(grey_lena_path, 'grey_result')
    apply_gaussian_laplacian(color_lena_path, 'color_result')
