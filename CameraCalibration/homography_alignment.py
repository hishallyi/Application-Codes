"""
@Author  ：Hishallyi
@Date    ：2024/4/18
@Code    : 利用单应变换进行图像对齐
"""


def homography_alignment(src_img_path, ref_img_path):
    """
    将源图像srcImg对齐到参考图像refImg上
    @param src_img_path: 源图像路径
    @param ref_img_path: 参考图像路径
    @return: 源图像和参考图像的单应矩阵Homography、对齐后的源图像
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    src_img = cv2.imread(src_img_path)
    ref_img = cv2.imread(ref_img_path)

    gray1 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # feature point matching
    detector = cv2.SIFT_create()
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # filter for matching points
    good_matches = []  # length：1004
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # extract the coordinates of the matching point
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # shape：(1004,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    #  calculate the homology matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # 3×3的单应矩阵

    # align the image and save
    aligned_img = cv2.warpPerspective(src_img, H, (src_img.shape[1], src_img.shape[0]))

    # displays the aligned image
    fig, axes = plt.subplots(1, 3, dpi=200)
    axes[0].imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Src Image')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Aligned Src Image')
    axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Reference Image')
    axes[2].axis('off')
    plt.tight_layout()
    # plt.savefig('ComparisonImage.jpg', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    return H, aligned_img


if __name__ == '__main__':
    import cv2

    H, aligned_img = homography_alignment(src_img_path="D:/FileDevelop/Datasets/Galaxy/scene_2/image_3.jpg",
                                          ref_img_path="D:/FileDevelop/Datasets/Galaxy/scene_2/image_1.jpg")
    cv2.imwrite('D:/FileDevelop/Datasets/Galaxy/scene_2/alignedImage_3.jpg', aligned_img)
    print("Homography Matrix:", H)
