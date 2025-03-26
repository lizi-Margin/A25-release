import ffmpeg


def change_video_speed_and_fps(input_path, output_path, speed=1.2, output_fps=30):
    """
    调整无音频视频的速度和帧率

    Args:
        input_path (str): 输入视频文件路径
        output_path (str): 输出视频文件路径
        speed (float): 速度倍数，默认1.2（加快到1.2倍）
        output_fps (int): 输出帧率，默认30
    """
    try:
        # 检查输入文件
        probe = ffmpeg.probe(input_path)
        if not probe:
            raise FileNotFoundError(f"输入文件 {input_path} 不存在或无法读取")

        # 获取输入视频的原始帧率
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        input_fps = eval(video_stream['r_frame_rate'])  # 将字符串（如"25/1"）转为数值
        print(f"原始帧率: {input_fps} fps")

        # 读取输入视频
        stream = ffmpeg.input(input_path)

        # 设置输出参数：调整速度和帧率（无音频）
        stream = ffmpeg.output(
            stream,
            output_path,
            vf=f'setpts={1 / speed:.6f}*PTS',  # 设置视频速度
            r=output_fps,  # 设置输出帧率
            **{'c:v': 'libx264'},  # 视频编码器
            preset='medium',  # 编码速度/质量权衡
            **{'-an': None},  # 禁用音频（修正为 -an）
            map='0:v',  # 明确映射视频流
            **{'y': None}  # 强制覆盖输出文件（-y）
        )

        # 执行转换
        ffmpeg.run(stream, quiet=True)
        print(f"视频速度已变为原来的{speed}倍，帧率从 {input_fps} 改为 {output_fps} fps，保存为 {output_path}")

    except ffmpeg.Error as e:
        error_message = e.stderr.decode('utf-8') if e.stderr else "未知错误"
        print('发生错误:', error_message)
    except FileNotFoundError as e:
        print('发生错误:', str(e))
    except Exception as e:
        print('发生未知错误:', str(e))


# 使用示例
if __name__ == "__main__":
    input_file = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/无锡低温烟雾环境双光视频/output_tr.mp4"
    output_file = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/无锡低温烟雾环境双光视频/s_output_tr.mp4"
    change_video_speed_and_fps(input_file, output_file, speed=1.2, output_fps=30)