import numpy as np
from PIL import Image

def bilinear_interpolation(image, scale_factor):
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
            src_x = x / scale_factor
            src_y = y / scale_factor

            # 原始像素坐标
            x0 = int(src_x)
            y0 = int(src_y)
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)

            # 双线性插值
            dx = src_x - x0
            dy = src_y - y0
            pixel_value_x0y0 = image.getpixel((x0, y0))
            pixel_value_x1y0 = image.getpixel((x1, y0))
            pixel_value_x0y1 = image.getpixel((x0, y1))
            pixel_value_x1y1 = image.getpixel((x1, y1))

            # 分别对每个通道进行插值计算
            r = (1 - dx) * (1 - dy) * pixel_value_x0y0[0] + dx * (1 - dy) * pixel_value_x1y0[0] + \
                (1 - dx) * dy * pixel_value_x0y1[0] + dx * dy * pixel_value_x1y1[0]
            g = (1 - dx) * (1 - dy) * pixel_value_x0y0[1] + dx * (1 - dy) * pixel_value_x1y0[1] + \
                (1 - dx) * dy * pixel_value_x0y1[1] + dx * dy * pixel_value_x1y1[1]
            b = (1 - dx) * (1 - dy) * pixel_value_x0y0[2] + dx * (1 - dy) * pixel_value_x1y0[2] + \
                (1 - dx) * dy * pixel_value_x0y1[2] + dx * dy * pixel_value_x1y1[2]

            new_image.putpixel((x, y), (int(r), int(g), int(b)))

    return new_image

# 加载图像
image = Image.open("girl.jpg")

# 缩放因子
scale_factor = 2

# 进行双线性插值
new_image = bilinear_interpolation(image, scale_factor)

# 保存插值后的图像
new_image.save("bilinear_output.jpg")
