import subprocess
import os
import cv2
import sys

def convert_video_fps(input_file, output_file, target_fps=30):
    """
    将视频转换为指定的帧率，同时保持原始时长
    
    参数:
        input_file (str): 输入视频文件路径
        output_file (str): 输出视频文件路径
        target_fps (float): 目标帧率
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
    
    # 获取视频信息
    print(f"尝试打开视频文件...")
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"错误: 无法使用OpenCV打开视频文件: '{input_file}'")
        
        # 尝试使用FFmpeg获取视频信息
        try:
            print("尝试使用FFmpeg获取视频信息...")
            cmd = ['ffmpeg', '-i', input_file]
            result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            print(f"FFmpeg输出:\n{result.stderr.decode()}")
        except Exception as e:
            print(f"使用FFmpeg获取视频信息失败: {e}")
        
        return False
    
    # 获取原始视频信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {original_fps} fps, {frame_count} 帧")
    
    if original_fps <= 0 or frame_count <= 0:
        print("警告: 视频信息异常，可能无法正确读取视频")
    
    # 计算时长
    original_duration = frame_count / original_fps if original_fps > 0 else 0
    print(f"视频时长: {original_duration:.2f} 秒")
    
    # 使用FFmpeg进行转换
    speed_factor = original_fps / target_fps if original_fps > 0 else 1.0
    print(f"速度因子: {speed_factor}")
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f'setpts={speed_factor}*PTS',
        '-r', str(target_fps),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
    ]
    
    # 检查是否有音频流
    audio_check_cmd = ['ffmpeg', '-i', input_file, '-hide_banner']
    audio_check = subprocess.run(audio_check_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    has_audio = "Audio:" in audio_check.stderr.decode()
    
    if has_audio:
        print("检测到视频包含音频，添加音频处理参数")
        ffmpeg_cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-af', f'atempo={1/speed_factor}'
        ])
    else:
        print("未检测到音频流")
        ffmpeg_cmd.append('-an')
    
    ffmpeg_cmd.append(output_file)
    
    print("执行FFmpeg命令:", " ".join(ffmpeg_cmd))
    
    try:
        process = subprocess.Popen(
            ffmpeg_cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
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
            print(f"输出文件大小: {filesize/1024/1024:.2f} MB")
            
            # 验证新视频
            new_cap = cv2.VideoCapture(output_file)
            if new_cap.isOpened():
                new_fps = new_cap.get(cv2.CAP_PROP_FPS)
                new_frame_count = int(new_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                new_duration = new_frame_count / new_fps if new_fps > 0 else 0
                
                print(f"新视频: {new_fps} fps, {new_frame_count} 帧")
                print(f"原始时长: {original_duration:.2f} 秒, 新时长: {new_duration:.2f} 秒")
                new_cap.release()
            else:
                print("警告: 无法使用OpenCV打开新视频进行验证")
        else:
            print(f"警告: 输出文件不存在: '{output_file}'")
        
        return True
    
    except Exception as e:
        print(f"转换过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        cap.release()

if __name__ == "__main__":
    print("--- 视频帧率转换工具 ---")
    print(f"Python版本: {sys.version}")
    print(f"OpenCV版本: {cv2.__version__}")
    print(f"当前工作目录: {os.getcwd()}")
    
    try:
        import argparse
        
        parser = argparse.ArgumentParser(description='将视频转换为指定帧率，保持原始时长')
        parser.add_argument('input', help='输入视频文件路径')
        parser.add_argument('output', help='输出视频文件路径')
        parser.add_argument('--fps', type=float, default=30.0, help='目标帧率 (默认: 30)')
        
        args = parser.parse_args()
        
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"目标帧率: {args.fps}")
        
        result = convert_video_fps(args.input, args.output, args.fps)
        if result:
            print("转换成功完成!")
        else:
            print("转换失败!")
            sys.exit(1)
            
    except Exception as e:
        print(f"程序出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)