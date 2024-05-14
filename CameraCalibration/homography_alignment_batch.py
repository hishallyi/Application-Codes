"""
@Author  ：Hishallyi
@Date    ：2024/4/23
@Code    : 相邻图像对齐
"""


def homography_alignment_batch(image_folder):
    """
    批量对齐图像
    :param image_folder: 图像文件夹路径
    @return: 对齐后的图像列表和单应矩阵列表，对齐图像保存在image_folder下，文件名为aligned_image_i.jpg
    """
    import cv2
    import numpy as np
    import glob

    image_paths = glob.glob(f"{image_folder}/*.jpg")
    # Read images and turn to grayscale map
    images = [cv2.imread(path) for path in image_paths]
    # turn to grayscale map
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    # select the feature detector
    feature_detector = cv2.SIFT_create()

    # find feature points and corresponding descriptors
    keypoints_list = []
    descriptors_list = []
    for image in gray_images:
        keypoints, descriptors = feature_detector.detectAndCompute(image, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # match feature points
    matcher = cv2.BFMatcher()
    matches = []
    for i in range(len(gray_images) - 1):
        matches.append(matcher.match(descriptors_list[i], descriptors_list[i + 1]))

    # calculate the transformation matrix
    transforms = []
    for i, match in enumerate(matches):
        src_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # 5.0 is the threshold of RANSAC
        transforms.append(H)

    # 对图像进行重采样
    aligned_images = [images[0]]
    for i in range(len(gray_images) - 1):
        aligned_image = cv2.warpPerspective(images[i + 1], transforms[i], (images[i].shape[1], images[i].shape[0]))
        aligned_images.append(aligned_image)

    # 保存对齐后的图像
    for i, aligned_image in enumerate(aligned_images):
        cv2.imwrite(f"{image_folder}/aligned_image_" + str(i) + ".jpg", aligned_image)

    return aligned_images, transforms


if __name__ == '__main__':

    image_folder = "D:/FileDevelop/Datasets/Galaxy/scene_1"
    aligned_images, transforms = homography_alignment_batch(image_folder)
    for i, transform in enumerate(transforms):
        print(f"Transform matrix of image {i + 1} and {i + 2}: \n{transform}")
