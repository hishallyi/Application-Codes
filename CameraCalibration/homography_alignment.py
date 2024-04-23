"""
@Author  ：Hishallyi
@Date    ：2024/4/18
@Code    : 利用单应变换进行图像对齐
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def homography_alignment(src_img, ref_img):
    """
    将源图像srcImg对齐到参考图像refImg上
    @param src_img: 源图像的RGB三通道像素值
    @param ref_img: 参考图像的RGB三通道像素值
    @return: 源图像和参考图像的单应矩阵Homography
    """
    gray1 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # 提取特征点和计算特征描述符
    detector = cv2.SIFT_create()
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    # 特征点匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 筛选匹配点
    good_matches = []  # length：1004
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # shape：(1004,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # 3×3的单应矩阵

    # 对齐图像
    aligned_img = cv2.warpPerspective(src_img, H, (src_img.shape[1], src_img.shape[0]))

    cv2.imwrite('aligned_img2.jpg', cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))

    # 显示对齐后的图像
    fig, axes = plt.subplots(1, 3, dpi=200)

    axes[0].imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image 1')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Aligned Image 1')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Image 2')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('images\\homographyAlignment.jpg', bbox_inches='tight', pad_inches=0.1)
    plt.show()
