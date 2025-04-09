import sys
import os, cv2
import subprocess
import numpy as np
from tqdm import tqdm

from A25.video_utils import _ensure_vid_format
from A25.shape import get_shaped_vid_path
from A25.time_alignment import get_aligned_vid_path as get_time_aligned_vid_path
from A25.global_config import GlobalConfig as cfg
from A25.cache import with_dual_cache

def convert_vid_fps_with_speed(input_file,original_fps=25, target_fps=30):
    """
    将视频从原始帧率转换为目标帧率，允许视频时长和播放速度改变

    参数:
        input_file (str): 输入视频文件路径
        original_fps (int): 原始帧率 (默认: 25)
        target_fps (int): 目标帧率 (默认: 30)
    """
    output_file = f"{cfg.outputdir}/output_convert_video_fps_with_speed.mp4"
    print(f"开始转换视频: '{input_file}' -> '{output_file}'")

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: '{input_file}'")
        return input_file

    # 检查输出路径是否可写
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            print(f"警告: 输出目录不存在: '{output_dir}'")
            os.makedirs(output_dir, exist_ok=True)
            print(f"已创建输出目录: '{output_dir}'")

        # 测试输出路径是否可写
        test_file = os.path.join(output_dir, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("输出目录可写")
    except Exception as e:
        print(f"错误: 无法写入输出目录: {e}")
        return input_file

    # 检查FFmpeg是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("FFmpeg 可用")
    except Exception as e:
        print(f"错误: FFmpeg不可用: {e}")
        print("请确保FFmpeg已安装并添加到PATH")
        return input_file

    # 计算播放速度因子
    speed_factor = target_fps / original_fps
    print(f"播放速度因子: {speed_factor:.2f} (视频将加速 {speed_factor:.2f} 倍)")

    # 构建FFmpeg命令
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_file,  # 输入文件
        '-vf', f"setpts={1/speed_factor}*PTS",  # 调整播放速度
        '-r', str(target_fps),  # 设置目标帧率
        '-c:v', 'libx264',  # 视频编码器
        '-preset', 'medium',  # 编码速度与质量平衡
        '-crf', '18',  # 视频质量 (0-51, 越小质量越高)
    ]

    # 检查是否有音频流
    audio_check_cmd = ['ffmpeg', '-i', input_file, '-hide_banner']
    audio_check = subprocess.run(audio_check_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    has_audio = "Audio:" in audio_check.stderr.decode('utf-8', errors='ignore')

    if has_audio:
        print("检测到视频包含音频，添加音频处理参数")
        ffmpeg_cmd.extend([
            '-c:a', 'aac',  # 音频编码器
            '-b:a', '192k',  # 音频比特率
            '-af', f'atempo={speed_factor}'  # 调整音频速度
        ])
    else:
        print("未检测到音频流")
        ffmpeg_cmd.append('-an')  # 禁用音频

    ffmpeg_cmd.append('-y')  # 确认覆盖 
    ffmpeg_cmd.append(output_file)  # 输出文件

    print("执行FFmpeg命令:", " ".join(ffmpeg_cmd))

    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8'
        )

        print("FFmpeg正在运行，请等待...")
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"FFmpeg错误 (代码 {process.returncode}):")
            print(stderr)
            return input_file

        print(f"转换成功! 视频已保存为 '{output_file}'")

        if os.path.exists(output_file):
            filesize = os.path.getsize(output_file)
            print(f"输出文件大小: {filesize / 1024 / 1024:.2f} MB")
        else:
            print(f"警告: 输出文件不存在: '{output_file}'")

        return output_file

    except Exception as e:
        print(f"转换过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        return input_file

def convert_vid_fps(input_file, target_fps=25.0):
    output_file = f"{cfg.outputdir}/output_convert_fps.mp4"
    # 打开视频文件
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"无法打开视频: {input_file}")
        return input_file

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
        return input_file

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
        return output_file
    else:
        print(f"转换失败，输出文件未创建或为空: {output_file}")
        return input_file

@with_dual_cache
def get_aligned_vid_path(wl_vid, ir_vid):
    ir_vid = convert_vid_fps_with_speed(ir_vid, original_fps=25, target_fps=30)
    ir_vid = convert_vid_fps(ir_vid, target_fps=25)
    wl_vid, ir_vid = get_shaped_vid_path(wl_vid, ir_vid)
    wl_vid, ir_vid = get_time_aligned_vid_path(wl_vid, ir_vid)
    return _ensure_vid_format(wl_vid), _ensure_vid_format(ir_vid)