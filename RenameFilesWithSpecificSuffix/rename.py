import os
import glob


def rename_files(img_folder, file_ext, new_name_prefix):
    """
    对指定后缀的文件进行批量改名
    @param img_folder: 改名文件所在目录的path
    @param file_ext: 改名文件的后缀名
    @param new_name_prefix: 改名的文件前缀
    @return:
    """
    images = glob.glob(f"{img_folder}/*{file_ext}")

    for idx, img_path in enumerate(images):
        img_name, img_ext = os.path.splitext(os.path.basename(img_path))
        new_img_name = f"{new_name_prefix}{idx + 1}{img_ext}"
        new_img_path = os.path.join(img_folder, new_img_name)
        os.rename(img_path, new_img_path)
        print(f"Renamed {img_path} to {new_img_path}")


if __name__ == '__main__':
    img_folder = "../CameraCalibration/images/datasets"  # 指定修改的目录
    file_ext = ".jpg"  # 指定文件后缀名
    new_name_prefix = "image00"

    rename_files(img_folder, file_ext, new_name_prefix)
