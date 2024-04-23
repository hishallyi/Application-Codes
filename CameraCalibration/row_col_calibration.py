"""
@Author  ：Hishallyi
@Date    ：2024/4/23
@Code    : 行列对齐的样例
"""

import cv2
import numpy as np


def row_calibration(images):
    # 选择特征点检测器
    feature_detector = cv2.SIFT_create()

    # 寻找特征点和对应描述符
    keypoints_list = []
    descriptors_list = []
    for image in images:
        keypoints, descriptors = feature_detector.detectAndCompute(image, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # 匹配特征点
    matcher = cv2.BFMatcher()
    matches = []
    for i in range(len(images) - 1):
        matches.append(matcher.match(descriptors_list[i], descriptors_list[i + 1]))

    # 计算变换矩阵
    transforms = []
    for match in matches:
        src_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        transforms.append(M)

    # 对图像进行重采样
    aligned_images = [images[0]]
    for i in range(len(images) - 1):
        aligned_image = cv2.warpPerspective(images[i + 1], transforms[i], (images[i].shape[1], images[i].shape[0]))
        aligned_images.append(aligned_image)

    return aligned_images


def column_calibration(images):
    # 在这里实现列校准的代码
    pass


# 读取图像
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
images = [cv2.imread(path) for path in image_paths]

# 行校准
aligned_images_row = row_calibration(images)

# 列校准
# aligned_images_column = column_calibration(aligned_images_row)
