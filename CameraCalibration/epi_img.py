"""
@Author  ：Hishallyi
@Date    ：2024/4/28
@Code    :
"""
import numpy as np
import matplotlib.pyplot as plt


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


# 假设你已经对图像进行了对齐，并且有了对齐后的图像列表 aligned_images
# 这里假设 aligned_images 是一个包含对齐后图像的二维数组的列表
# 每个图像都是一个二维数组，表示一个灰度图像

# 创建 EPI 图像
# epi = create_epi(aligned_images)

# 可视化 EPI 图像
# visualize_epi(epi)
