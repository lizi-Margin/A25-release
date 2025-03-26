import cv2
import numpy as np
import os
from tqdm import tqdm

def convert_fps(input_file, output_file, target_fps=25.0):
    # 打开视频文件
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"无法打开视频: {input_file}")
        return False

    # 获取源视频信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 计算视频总时长(秒)
    original_duration = total_frames / original_fps

    # 计算新视频需要的总帧数
    new_total_frames = int(original_duration * target_fps)

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4V 编解码器
    out = cv2.VideoWriter(output_file, fourcc, target_fps, (width, height))

    if not out.isOpened():
        print("无法创建输出视频文件，请检查路径或编解码器!")
        cap.release()
        return False

    # 创建帧采样索引
    original_timestamps = np.linspace(0, original_duration, total_frames, endpoint=False)
    new_timestamps = np.linspace(0, original_duration, new_total_frames, endpoint=False)

    frame_indices = [np.abs(original_timestamps - ts).argmin() for ts in new_timestamps]

    print(f"原始视频: {original_fps} FPS, {total_frames} 帧, 时长 {original_duration:.2f} 秒")
    print(f"目标视频: {target_fps} FPS, {new_total_frames} 帧, 时长 {original_duration:.2f} 秒")

    # 读取并写入视频帧
    with tqdm(total=new_total_frames, desc="处理帧") as pbar:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"读取帧 {frame_idx} 失败，提前结束!")
                break
            out.write(frame)
            pbar.update(1)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 验证输出文件
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"转换完成! 输出文件: {output_file}")
        return True
    else:
        print(f"转换失败，输出文件未创建或为空: {output_file}")
        return False

if __name__ == "__main__":
    input_file = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/speed_output_tr_smoked3.mp4"
    output_file = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/speed_output_tr_smoked3_25fps.mp4"
    target_fps = 25.0  # 设置目标帧率为 25fps

    if not os.path.isfile(input_file):
        print(f"输入文件不存在: {input_file}")
        exit(1)

    success = convert_fps(input_file, output_file, target_fps)
    print("视频转换成功完成!" if success else "视频转换失败!")