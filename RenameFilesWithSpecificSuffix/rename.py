import os
import glob

"""
需要修改的三个点：
- 改名文件所在文件夹的位置：img_folder
- 改名文件的后缀名：file_ext
- 新文件名的格式：new_img_name
"""
# 指定修改的目录和文件后缀名
img_folder = ""
file_ext = ".jpg"

images = glob.glob(f"{img_folder}/*{file_ext}")   # 获取以jpg为后缀的文件目录

# 遍历每个图像文件
for idx, img_path in enumerate(images):
    img_name, img_ext = os.path.splitext(os.path.basename(img_path))   # img_ext保存文件扩展名
    new_img_name = f"image00{idx + 1}{img_ext}"   # 定义新文件名格式
    new_img_path = os.path.join(img_folder, new_img_name)
    
    # 重命名图像文件
    os.rename(img_path, new_img_path)
    
    print(f"Renamed {img_path} to {new_img_path}")
