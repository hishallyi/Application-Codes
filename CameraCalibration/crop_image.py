"""
@Author  ：Hishallyi
@Date    ：2024/4/22
@Code    : 裁剪图像
"""

import cv2

# 读取图像
image = cv2.imread('input_image.jpg')

# 获取图像尺寸
height, width = image.shape[:2]

# 定义裁剪区域
top_left = (width // 4, height // 4)
bottom_right = (width * 3 // 4, height * 3 // 4)

# 裁剪图像
cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# 显示裁剪后的图像
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
