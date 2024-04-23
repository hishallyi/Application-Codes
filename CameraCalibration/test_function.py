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
camera_path = "images"
get_camera_params(camera_path=camera_path, board_rows=7, board_cols=10, square_size_mm=21)
