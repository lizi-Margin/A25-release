import ffmpeg


def resize_video(input_path, output_path, width=640, height=512):
    try:
        # 读取输入视频
        stream = ffmpeg.input(input_path)

        # 设置输出参数并调整分辨率
        stream = ffmpeg.output(
            stream,
            output_path,
            vf=f'scale={width}:{height}',  # 设置缩放尺寸
            **{'c:v': 'libx264'},  # 视频编码器（使用正确语法）
            preset='medium',  # 编码速度/质量权衡
            **{'c:a': 'aac'},  # 音频编码器（使用正确语法）
            **{'b:a': '128k'}  # 音频比特率
        )

        # 执行转换
        ffmpeg.run(stream)
        print(f"视频已成功转换为 {width}x{height}，保存为 {output_path}")

    except ffmpeg.Error as e:
        # 添加对 stderr 是否为 None 的检查
        error_message = e.stderr.decode('utf-8') if e.stderr else "未知错误"
        print('发生错误:', error_message)
# 使用示例
if __name__ == "__main__":
    input_file = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/无锡低温烟雾环境双光视频/output_tr.mp4"  # 输入视频文件名
    output_file = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/无锡低温烟雾环境双光视频/2output_tr.mp4"  # 输出视频文件名
    resize_video(input_file, output_file)