import numpy as np
from PIL import Image

def nearest_neighbor_interpolation(image, scale_factor):
    # 原始图像的尺寸
    width, height = image.size

    # 新图像的尺寸
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # 创建新的图像对象
    new_image = Image.new("RGB", (new_width, new_height))

    # 计算每个新像素对应的原始像素位置，并进行插值
    for x in range(new_width):
        for y in range(new_height):
            src_x = int(x / scale_factor)
            src_y = int(y / scale_factor)
            new_image.putpixel((x, y), image.getpixel((src_x, src_y)))

    return new_image

# 加载图像
image = Image.open("girl.jpg")

# 缩放因子
scale_factor = 2

# 进行最近邻插值
new_image = nearest_neighbor_interpolation(image, scale_factor)

# 保存插值后的图像
new_image.save("nearest_neighbor_output.jpg")
