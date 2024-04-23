"""
@Author  ：Hishallyi
@Date    ：2024/4/23
@Code    : 相机标定实例
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# params of Calibration board
board_rows = 7
board_cols = 10
square_size_mm = 21

# define World coordinate
objp = np.zeros((board_rows * board_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
objp *= square_size_mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Get calibration images
images = glob.glob('images\\*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (board_cols, board_rows), None)

    if ret == True:
        objpoints.append(objp)

        # Subpixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        if [corners2]:
            imgpoints.append(corners2)
        else:
            imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (board_cols, board_rows), corners, ret)
        write_name = 'corners_found'+str(idx)+'.jpg'
        cv2.imwrite(write_name, img)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()

# Test undistortion on an image
img = cv2.imread('images\\image005.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save the camera calibration result
calibration_result = {}
calibration_result["IntrinsicMatrix"] = mtx  # 内参矩阵
calibration_result["dist_coeffs"] = dist  # 畸变系数distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
calibration_result['rvecs'] = rvecs  # 旋转向量
calibration_result['tvecs'] = tvecs  # 平移向量
calibration_result['objpoints'] = objpoints  # 标定板上的角点的世界坐标
calibration_result['imgpoints'] = imgpoints  # 标定板在图像上的像素坐标
calibration_result['image_size'] = img_size  # 图像尺寸
pickle.dump(calibration_result, open("images/wide_dist_pickle.p", "wb"))


# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

def ImageUndistortion(image_path, save_path, mtx, dist):
    img = cv2.imread(image_path)
    dst = cv2.undistort(src=img,
                        cameraMatrix=mtx,
                        distCoeffs=dist,
                        newCameraMatrix=None)
    cv2.imwrite(save_path, dst)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)


ImageUndistortion('images\image005.jpg', 'images\image005_undistort.jpg', mtx, dist)

# 初始化重投影误差列表
reprojection_errors = []

# 遍历每张图像
for i in range(len(objpoints)):
    # using calibration result to reproject
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

    # compute reprojection error for each image pair
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    reprojection_errors.append(error)

    print(f"Image {i + 1} Reprojection Error: {error}")

# compute average reproject error
mean_reprojection_error = np.mean(reprojection_errors)
print(f"\nMean Reprojection Error: {mean_reprojection_error}")
