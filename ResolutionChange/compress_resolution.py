"""
@Author  ：Hishallyi
@Date    ：2024/6/25
@Code    : 压缩分辨率
"""
from PIL import Image

# 打开原始图像
original_image_path = '原图.jpg'
image = Image.open(original_image_path)

# 原始图像尺寸
original_width, original_height = image.size

# TODO: 设置目标宽高
target_width, target_height = 144, 156
target_ratio = target_width / target_height

# 计算裁剪区域
original_ratio = original_width / original_height

if original_ratio > target_ratio:
    # 原图宽高比大于目标宽高比，裁剪宽度
    new_width = int(original_height * target_ratio)
    new_height = original_height
    offset = (original_width - new_width) // 2
    box = (offset, 0, offset + new_width, new_height)
else:
    # 原图宽高比小于目标宽高比，裁剪高度
    new_width = original_width
    new_height = int(original_width / target_ratio)
    offset = (original_height - new_height) // 2
    box = (0, offset, new_width, offset + new_height)

# 裁剪图像
cropped_image = image.crop(box)

# 调整图像大小
resized_image = cropped_image.resize((target_width, target_height), Image.LANCZOS)

# 保存调整后的图像
resized_image_path = '压缩后图像.jpg'
resized_image.save(resized_image_path)

print(f'Resized image saved to {resized_image_path}')
