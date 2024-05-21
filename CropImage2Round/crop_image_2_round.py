"""
@Author  ：Hishallyi
@Date    ：2024/5/22
@Code    : 将图片裁剪成圆形
"""
from PIL import Image, ImageDraw


def crop_to_circle(image_path, output_path):
    # 打开图像文件
    with Image.open(image_path) as img:
        # 确保图像是正方形
        width, height = img.size
        min_dimension = min(width, height)

        # 计算裁剪区域
        left = (width - min_dimension) / 2
        top = (height - min_dimension) / 2
        right = (width + min_dimension) / 2
        bottom = (height + min_dimension) / 2

        # 裁剪图像为正方形
        img = img.crop((left, top, right, bottom))

        # 创建一个新的图像，白色背景
        mask = Image.new('L', (min_dimension, min_dimension), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, min_dimension, min_dimension), fill=255)

        # 创建带有透明背景的新图像
        result = Image.new('RGBA', (min_dimension, min_dimension))
        result.paste(img, (0, 0), mask)

        # 保存结果，确保保存为PNG格式
        result.save(output_path, format='PNG')


if __name__ == '__main__':
    input_image_path = 'logo.jpg'  # 输入图片路径
    output_image_path = 'output_image.jpg'  # 输出图片路径
    crop_to_circle(input_image_path, output_image_path)
