"""
@Author  ：Hishallyi
@Date    ：2024/4/23
@Code    : 图像仿射变换
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取两个图像
image1 = cv2.imread('images\\image002.jpg')
image2 = cv2.imread('images\\image003.jpg')

# 提取特征点和计算特征描述符
detector = cv2.SIFT_create()
kp1, des1 = detector.detectAndCompute(image1, None)
kp2, des2 = detector.detectAndCompute(image2, None)

# 特征点匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 筛选匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 提取匹配点的坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 计算仿射变换矩阵
M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

# 对齐图像
aligned_image1 = cv2.warpAffine(image1, M, (image1.shape[1], image1.shape[0]))

# 显示对齐后的图像
# cv2.imshow('Aligned Image 1', aligned_image1)
# cv2.imshow('Image 2', image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# 在每个子图中显示不同的图像
axes[0].imshow(image1)
axes[0].set_title('Image 1')
axes[0].axis('off')

axes[1].imshow(aligned_image1)
axes[1].set_title('aligned_image1')
axes[1].axis('off')

axes[2].imshow(image2)
axes[2].set_title('Image 2')
axes[2].axis('off')

plt.tight_layout()  # 调整子图布局，防止重叠
plt.savefig('combined_images.jpg', bbox_inches='tight', pad_inches=0.1)
plt.show()
