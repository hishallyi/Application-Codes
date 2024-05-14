"""
@Author  ：Hishallyi
@Date    ：2024/4/28
@Code    :
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2


def create_epi(aligned_images):
    """
    创建光场 EPI 图像，EPI全称Epipolar Plane Image，它是光场空-角域耦合关系的一种降维表现形式，其斜率表示视差信息。
    @param aligned_images: 对齐后的图像列表，每个图像应该是一个二维数组
    @return: EPI 图像
    """
    num_views = len(aligned_images)
    height, width = aligned_images[0].shape
    epi = np.zeros((height, num_views * width))

    for i, img in enumerate(aligned_images):
        epi[:, i * width:(i + 1) * width] = img

    return epi


def visualize_epi(epi):
    """
    可视化光场 EPI 图像
    @param epi: EPI 图像
    """
    plt.imshow(epi, cmap='gray')
    plt.title('Epipolar-plane image (EPI)')
    plt.xlabel('Viewpoint')
    plt.ylabel('Pixel row')
    plt.show()


if __name__ == '__main__':
    # 读取对齐图像
    aligned_images = [cv2.imread(path) for path in glob.glob("./images/datasets/aligned_image_*.jpg")]
    aligned_images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in aligned_images]

    # 创建 EPI 图像
    epi = create_epi(np.array(aligned_images_gray))

    # 可视化 EPI 图像
    visualize_epi(epi)
