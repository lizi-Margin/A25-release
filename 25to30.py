import subprocess
import os
import sys

def convert_video_fps(input_file, output_file, original_fps=25, target_fps=30):
    """
    将视频从原始帧率转换为目标帧率，允许视频时长和播放速度改变

    参数:
        input_file (str): 输入视频文件路径
        output_file (str): 输出视频文件路径
        original_fps (int): 原始帧率 (默认: 25)
        target_fps (int): 目标帧率 (默认: 30)
    """
    print(f"开始转换视频: '{input_file}' -> '{output_file}'")

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: '{input_file}'")
        return False

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
        return False

    # 检查FFmpeg是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("FFmpeg 可用")
    except Exception as e:
        print(f"错误: FFmpeg不可用: {e}")
        print("请确保FFmpeg已安装并添加到PATH")
        return False

    # 计算播放速度因子
    speed_factor = target_fps / original_fps
    print(f"播放速度因子: {speed_factor:.2f} (视频将加速 {speed_factor:.2f} 倍)")

    # 构建FFmpeg命令
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_file,  # 输入文件
        '-vf', f'setpts={1/speed_factor}*PTS',  # 调整播放速度
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
            return False

        print(f"转换成功! 视频已保存为 '{output_file}'")

        if os.path.exists(output_file):
            filesize = os.path.getsize(output_file)
            print(f"输出文件大小: {filesize / 1024 / 1024:.2f} MB")
        else:
            print(f"警告: 输出文件不存在: '{output_file}'")

        return True

    except Exception as e:
        print(f"转换过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("--- 视频帧率转换工具 ---")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")

    # 设置路径和参数
    input_file = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/output_tr_smoked3.mp4"   # 输入视频文件路径
    output_file = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/speed_output_tr_smoked3.mp4"   # 输出视频文件路径
    original_fps = 25  # 原始帧率
    target_fps = 30  # 目标帧率

    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"原始帧率: {original_fps}")
    print(f"目标帧率: {target_fps}")

    # 调用转换函数
    result = convert_video_fps(input_file, output_file, original_fps, target_fps)
    if result:
        print("转换成功完成!")
    else:
        print("转换失败!")
        sys.exit(1)