"""
@Author  ：Hishallyi
@Date    ：2024/4/13
@Homepage：https://github.com/Hishllyi
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image = cv2.imread('experiment.jpg', cv2.IMREAD_GRAYSCALE)

# 将图像转换为浮点数类型
f_image = np.float32(image)

# 进行二维傅立叶变换
f_transform = cv2.dft(f_image, flags=cv2.DFT_COMPLEX_OUTPUT)

# 将零频率分量移到图像中心
f_shift = np.fft.fftshift(f_transform)

# 计算幅值谱
magnitude_spectrum = 20 * np.log(cv2.magnitude(f_shift[:, :, 0], f_shift[:, :, 1]))

# 显示原始图像和频谱图像
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
