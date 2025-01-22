import cv2
from matplotlib import pyplot as plt


def sift_feature_matching(img_path1, img_path2):
    """
    SIFT特征匹配

    :param img_path1: 图片1的路径
    :param img_path2: 图片2的路径
    """
    # 读取图像
    img1 = cv2.imread(img_path1)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 创建SIFT对象
    sift = cv2.xfeatures2d.SIFT_create()

    # 获取图片1的关键点和描述符
    kp1, des1 = sift.detectAndCompute(gray1, None)

    # 读取图像2
    img2 = cv2.imread(img_path2)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 获取图片2的关键点和描述符
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 使用Brute-Force匹配器进行匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 根据Lowe's ratio进行筛选
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # 绘制匹配结果
    result_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)

    # 创建并展示图像
    plt.figure(figsize=(15, 5), dpi=300)
    plt.imshow(result_image)
    plt.savefig('sift_feature_matching_result.png')
    plt.show()


if __name__ == '__main__':
    sift_feature_matching('images/mountain1.png', 'images/mountain2.png')
