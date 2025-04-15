import cv2
import os


def vid2img(video_path, N=1):
    if not os.path.exists(video_path):
        print(f"视频文件 {video_path} 不存在。")
        return

    output_dir = f"{video_path}.d"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件 {video_path}。")
        return

    frame_count = 0
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % N == 0:
            # 保存帧为图片
            frame_filename = os.path.join(output_dir,
                                          f"frame_{save_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            save_count += 1

        frame_count += 1

    cap.release()
    print(f"共抽取并保存了 {save_count} 帧图片到 {output_dir} 目录。")


if __name__ == '__main__':
    vid2img("./tmp/对齐video/output_fused_smoked2.mp4", N=1)
