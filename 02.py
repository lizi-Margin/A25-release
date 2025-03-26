import ffmpeg


def adjust_dar(input_path, output_path, target_dar="16:9"):
    """
    调整视频的显示宽高比 (DAR)，保持分辨率不变

    Args:
        input_path (str): 输入视频文件路径
        output_path (str): 输出视频文件路径
        target_dar (str): 目标显示宽高比，例如 "16:9", "5:4", 或 "1:1"
    """
    try:
        # 检查输入文件
        probe = ffmpeg.probe(input_path)
        if not probe:
            raise FileNotFoundError(f"输入文件 {input_path} 不存在或无法读取")

        # 读取输入视频
        stream = ffmpeg.input(input_path)

        # 设置输出参数
        stream = ffmpeg.output(
            stream,
            output_path,
            vf=f'setdar={target_dar}',  # 设置目标 DAR
            **{
                'c:v': 'libx264',  # 视频编码器
                # '-an': None,  # 禁用音频
                # '-y': None  # 覆盖输出文件
            }
        )

        # 执行转换
        ffmpeg.run(stream, quiet=True)
        print(f"已将视频的 DAR 调整为 {target_dar}，保存为 {output_path}")

    except ffmpeg.Error as e:
        error_message = e.stderr.decode('utf-8') if e.stderr else "未知错误"
        print('发生错误:', error_message)
    except FileNotFoundError as e:
        print('发生错误:', str(e))


# 使用示例
if __name__ == "__main__":
    input_rgb = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/无锡低温烟雾环境双光视频/2output_rgb.mp4"
    input_tr = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/无锡低温烟雾环境双光视频/2output_tr.mp4"
    output_rgb = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/无锡低温烟雾环境双光视频/2output_rgb_adjusted.mp4"
    output_tr = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/无锡低温烟雾环境双光视频/2output_tr_adjusted.mp4"

    # 统一为 16:9（长方形）
    # adjust_dar(input_rgb, output_rgb, target_dar="16:9")
    # adjust_dar(input_tr, output_tr, target_dar="16:9")

    # 可选：统一为 5:4（接近正方形）
    adjust_dar(input_rgb, output_rgb, target_dar="5:4")
    adjust_dar(input_tr, output_tr, target_dar="5:4")

    # 可选：统一为 1:1（正方形）
    # adjust_dar(input_rgb, output_rgb, target_dar="1:1")
    # adjust_dar(input_tr, output_tr, target_dar="1:1")