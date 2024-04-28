"""
@Author  ：Hishallyi
@Date    ：2024/4/23
@Code    :
"""
import numpy as np
import cv2
import pickle
import glob
from get_camera_params import get_camera_params
from homography_alignment import homography_alignment

# camera_path = "images"
# get_camera_params(camera_path=camera_path, board_rows=7, board_cols=10, square_size_mm=21)

src_img = cv2.imread("images/datasets/image001.jpg")
ref_img = cv2.imread("images/datasets/image002.jpg")

H, aligned_img, aligned_img2 = homography_alignment(src_img, ref_img)
print("H:", H)
print("aligned_img:", aligned_img)
print("aligned_img2:", aligned_img2)
