import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm

def convert_fps(input_file, output_file, target_fps=25.0):
    """
    将视频从一个帧率转换到另一个帧率，同时保持原始时长
    使用帧采样方法实现
    
    参数:
    input_file: 输入视频文件路径
    output_file: 输出视频文件路径
    target_fps: 目标帧率，默认为25.0
    """
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以根据需要选择其他编解码器，如'XVID'
    out = cv2.VideoWriter(output_file, fourcc, target_fps, (width, height))
    
    # 创建帧采样索引
    # 使用线性插值确定哪些原始帧应该被采样到新视频中
    original_timestamps = np.linspace(0, original_duration, total_frames, endpoint=False)
    new_timestamps = np.linspace(0, original_duration, new_total_frames, endpoint=False)
    
    # 为每个新时间戳找到最近的原始帧
    frame_indices = []
    for ts in new_timestamps:
        # 找到最接近当前时间戳的原始帧索引
        idx = np.abs(original_timestamps - ts).argmin()
        frame_indices.append(idx)
    
    print(f"原始视频: {original_fps} FPS, {total_frames} 帧, 时长 {original_duration:.2f} 秒")
    print(f"目标视频: {target_fps} FPS, {new_total_frames} 帧, 时长 {original_duration:.2f} 秒")
    
    # 读取并写入视频帧
    with tqdm(total=new_total_frames, desc="处理帧") as pbar:
        for frame_idx in frame_indices:
            # 将视频指针设置到特定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # 写入帧到输出视频
            out.write(frame)
            pbar.update(1)
    
    # 释放资源
    cap.release()
    out.release()
    
    # 验证输出文件是否创建成功
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"转换完成! 输出文件: {output_file}")
        return True
    else:
        print(f"转换过程完成，但输出文件创建失败或为空: {output_file}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='视频帧率转换工具')
    parser.add_argument('input', type=str, help='输入视频文件路径')
    parser.add_argument('output', type=str, help='输出视频文件路径')
    parser.add_argument('--fps', type=float, default=25.0, help='目标帧率 (默认: 25.0)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.isfile(args.input):
        print(f"输入文件不存在: {args.input}")
        exit(1)
    
    # 执行转换
    success = convert_fps(args.input, args.output, args.fps)
    
    # 明确打印转换是否成功
    if success:
        print("视频转换成功完成!")
    else:
        print("视频转换失败!")