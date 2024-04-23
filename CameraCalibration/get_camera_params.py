"""
@Author  ：Hishallyi
@Date    ：2024/4/23
@Code    : 获取相机参数的工具函数
"""


def get_camera_params(camera_path, board_rows, board_cols, square_size_mm):
    """
    相机标定求相机参数
    @param camera_path: 存储单个相机视角的文件路径
    @param board_rows: 棋盘格行角点个数
    @param board_cols: 棋盘格列角点个数
    @param square_size_mm: 棋盘格正方形边长，单位：mm
    @return: 相机内参、旋转矩阵、平移矩阵、重投影误差
    """
    import numpy as np
    import cv2
    import pickle
    import glob
    import os

    # define World coordinate
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []  # The world coordinate system for all images
    img_points = []  # The pixel coordinate system of all images

    images = glob.glob(f'{camera_path}\\*.jpg')

    # Create a folder to save the corner image
    folder_path = f'{camera_path}\\corners_found_images'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (board_cols, board_rows), None)
        if ret:
            obj_points.append(objp)
            # Subpixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            # Draw checkerboard corner points
            cv2.drawChessboardCorners(img, (board_cols, board_rows), corners, ret)
            write_name = f'{folder_path}\\corners_found_' + str(idx) + '.jpg'
            cv2.imwrite(write_name, img)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    cv2.destroyAllWindows()
    img_get_size = cv2.imread(images[0])
    img_size = (img_get_size.shape[1], img_get_size.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rotation_matrix, translation_matrix = cv2.calibrateCamera(obj_points, img_points, img_size, None,
                                                                              None)

    print("Intrinsic Matrix：\n", mtx)
    print("Distortion coefficient：\n", dist)
    print("rotation matrix：\n", rotation_matrix)
    print("Translation matrix：\n", translation_matrix)

    # Save the camera calibration result
    calibration_result = {"IntrinsicMatrix": mtx,
                          "dist_coefficients": dist,
                          'rotation_matrix': rotation_matrix,
                          'trans_matrix': translation_matrix,
                          'obj_points': obj_points,
                          'img_points': img_points,
                          'image_size': img_size}
    pickle.dump(calibration_result, open(f"{camera_path}\\camera_params.p", "wb"))

    # Calculate the reprojection error for each image
    reprojection_errors = []
    for i in range(len(obj_points)):
        # using calibration result to reproject
        img_points2, _ = cv2.projectPoints(obj_points[i], rotation_matrix[i], translation_matrix[i], mtx, dist)
        # compute reprojection error for each image pair
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        reprojection_errors.append(error)
        print(f"Image {i + 1} Reprojection Error: {error}")

    # compute average reproject error
    mean_reprojection_error = np.mean(reprojection_errors)
    print(f"\nMean Reprojection Error: {mean_reprojection_error}")
