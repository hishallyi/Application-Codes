"""
@Author : Hishallyi
@Date   : 2024/7/5
@Code   : 用OpenCV通过插值改变图像分辨率
"""

import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('caoying.jpg')


# 将图像从 BGR 转换为 RGB（Matplotlib 使用 RGB）
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

width, height = 144, 156
# 定义新的尺寸（宽，高）
new_size = (width, height)

# 使用不同的插值方法来调整图像大小
resized_image_nearest = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
resized_image_linear = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
resized_image_cubic = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
resized_image_lanczos = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

# 创建一个画布来显示图像
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 显示每幅图像，并添加标签
axes[0, 0].imshow(resized_image_nearest)
axes[0, 0].set_title('Nearest Interpolation')
axes[0, 0].axis('off')  # 去除坐标轴

axes[0, 1].imshow(resized_image_linear)
axes[0, 1].set_title('Linear Interpolation')
axes[0, 1].axis('off')

axes[1, 0].imshow(resized_image_cubic)
axes[1, 0].set_title('Cubic Interpolation')
axes[1, 0].axis('off')

axes[1, 1].imshow(resized_image_lanczos)
axes[1, 1].set_title('Lanczos Interpolation')
axes[1, 1].axis('off')

# 显示图像
plt.tight_layout()
plt.show()

# 将调整后的图像保存到文件
# cv2.imwrite('resized_image.jpg', resized_image)

