import cv2 as cv
import numpy as np

def canny_edge_detection(image_path):
    # 第一步：高斯平滑滤波
    original_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    blurred_image = cv.GaussianBlur(original_image, (3, 3), 2)

    # 第二步：计算梯度
    gradient_x = cv.Sobel(blurred_image, cv.CV_64F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(blurred_image, cv.CV_64F, 0, 1, ksize=3)
    gradient_magnitude, gradient_direction = cv.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)

    # 第三步：非极大值抑制
    suppressed_image = cv.copyMakeBorder(gradient_magnitude, 1, 1, 1, 1, cv.BORDER_REPLICATE)
    for i in range(1, suppressed_image.shape[0] - 1):
        for j in range(1, suppressed_image.shape[1] - 1):
            theta = gradient_direction[i - 1, j - 1]
            if -22.5 <= theta < 22.5 or -157.5 <= theta <= -180 or 157.5 <= theta < 180:
                theta = 0
            elif 22.5 <= theta < 67.5 or -112.5 <= theta <= -157.5:
                theta = 45
            elif 67.5 <= theta < 112.5 or -67.5 <= theta <= -112.5:
                theta = 90
            elif 112.5 <= theta < 157.5 or -22.5 <= theta <= -67.5:
                theta = -45

            if (theta == 0 and suppressed_image[i, j] == np.max([suppressed_image[i, j], suppressed_image[i + 1, j], suppressed_image[i - 1, j]])) or \
               (theta == -45 and suppressed_image[i, j] == np.max([suppressed_image[i, j], suppressed_image[i - 1, j - 1], suppressed_image[i + 1, j + 1]])) or \
               (theta == 90 and suppressed_image[i, j] == np.max([suppressed_image[i, j], suppressed_image[i, j + 1], suppressed_image[i, j - 1]])) or \
               (theta == 45 and suppressed_image[i, j] == np.max([suppressed_image[i, j], suppressed_image[i - 1, j + 1], suppressed_image[i + 1, j - 1]])):
                suppressed_image[i, j] = gradient_magnitude[i - 1, j - 1]
            else:
                suppressed_image[i, j] = 0

    # 第四步：双阈值检测和边缘连接
    canny_image = np.zeros(suppressed_image.shape, dtype=np.uint8)
    low_threshold = 0.4 * np.max(suppressed_image)
    high_threshold = 0.5 * np.max(suppressed_image)

    for i in range(1, canny_image.shape[0] - 1):
        for j in range(1, canny_image.shape[1] - 1):
            if suppressed_image[i, j] < low_threshold:
                canny_image[i, j] = 0
            elif suppressed_image[i, j] > high_threshold:
                canny_image[i, j] = 255
            elif (suppressed_image[i + 1, j] < high_threshold or suppressed_image[i - 1, j] < high_threshold or
                  suppressed_image[i, j + 1] < high_threshold or suppressed_image[i, j - 1] < high_threshold or
                  suppressed_image[i - 1, j - 1] < high_threshold or suppressed_image[i - 1, j + 1] < high_threshold or
                  suppressed_image[i + 1, j + 1] < high_threshold or suppressed_image[i + 1, j - 1] < high_threshold):
                canny_image[i, j] = 255

    # 显示结果
    cv.imshow("Original Image", original_image)
    cv.imshow("Canny Edge Detection", canny_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    canny_edge_detection('data set/lena/lena512.bmp')
