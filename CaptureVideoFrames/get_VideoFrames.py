import cv2


def extract_frames(video_path, output_folder):
    # 读取视频
    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()
    count = 0

    # 提取视频中的所有帧
    while success:
        frame_filename = f"{output_folder}/frame_{count}.jpg"
        cv2.imwrite(frame_filename, image)  # 保存帧图像
        success, image = video_capture.read()
        print(f"Frame {count} extracted successfully")
        count += 1

    video_capture.release()


if __name__ == "__main__":
    video_path = "beauty.mp4"  # 视频文件路径
    output_folder = "frames"  # 保存帧图像的文件夹

    # 创建保存帧图像的文件夹
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 提取视频中的所有帧
    extract_frames(video_path, output_folder)
