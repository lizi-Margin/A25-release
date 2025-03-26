import gradio as gr
import cv2
import numpy as np
import os
import time
from sklearn.metrics import precision_score, recall_score
from skimage.metrics import structural_similarity as ssim
import pandas as pd


def merge_videos(video1_path, video2_path):
    start_time = time.time()

    # 读取两个视频
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # 获取视频属性
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "merged_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    # 用于计算指标的变量
    processed_frames = 0
    successful_merges = 0
    y_true = []
    y_pred = []
    psnr_values = []
    ssim_values = []
    frame_differences = []

    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            processed_frames += 1
            frame2 = cv2.resize(frame2, (width, height))

            try:
                merged_frame = np.hstack((frame1, frame2))
                out.write(merged_frame)
                successful_merges += 1

                y_true.append(1)
                y_pred.append(1)

                # 计算PSNR
                psnr_value = cv2.PSNR(frame1, frame2)
                psnr_values.append(psnr_value)

                # 计算SSIM
                frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                ssim_value = ssim(frame1_gray, frame2_gray)
                ssim_values.append(ssim_value)

                # 计算帧间差异
                diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
                frame_differences.append(diff)

            except Exception as e:
                y_true.append(1)
                y_pred.append(0)

    except Exception as e:
        return f"Error: {str(e)}", {}

    finally:
        cap1.release()
        cap2.release()
        out.release()

    # 计算指标
    end_time = time.time()
    processing_time = end_time - start_time

    accuracy = successful_merges / processed_frames if processed_frames > 0 else 0
    precision = precision_score(y_true, y_pred) if y_true else 0
    recall = recall_score(y_true, y_pred) if y_true else 0

    avg_psnr = np.mean(psnr_values) if psnr_values else 0
    avg_ssim = np.mean(ssim_values) if ssim_values else 0
    avg_frame_diff = np.mean(frame_differences) if frame_differences else 0
    frame_rate_consistency = min(total_frames1, total_frames2) / max(total_frames1, total_frames2)

    metrics = {
        "处理时间 (秒)": f"{processing_time:.2f}",
        "总帧数": processed_frames,
        "成功合并帧数": successful_merges,
        "准确率": f"{accuracy:.2%}",
        "精确率": f"{precision:.2%}",
        "召回率": f"{recall:.2%}",
        "平均PSNR (dB)": f"{avg_psnr:.2f}",
        "平均SSIM": f"{avg_ssim:.3f}",
        "平均帧间差异": f"{avg_frame_diff:.2f}",
        "帧率一致性": f"{frame_rate_consistency:.2%}",
        "输出文件大小 (MB)": f"{os.path.getsize(output_path) / 1024 / 1024:.2f}" if os.path.exists(
            output_path) else "N/A"
    }

    return output_path, metrics


# 样例指标数据
example_metrics = {
    "指标名称": [
        "处理时间 (秒)", "总帧数", "成功合并帧数", "准确率",
        "精确率", "召回率", "平均PSNR (dB)", "平均SSIM",
        "平均帧间差异", "帧率一致性", "输出文件大小 (MB)"
    ],
    "值": [
        "3.45", "120", "118", "98.33%",
        "100.00%", "98.33%", "35.67", "0.892",
        "12.34", "95.00%", "15.78"
    ],
    "说明": [
        "合并过程总耗时", "处理的视频帧总数", "成功合并的帧数", "成功合并比例",
        "合并成功的精度", "成功召回比例", "峰值信噪比（越高越好）", "结构相似性（0-1）",
        "帧间像素差异", "两视频长度匹配度", "输出视频大小"
    ]
}

# 转换为Dataframe以便在前端展示
example_df = pd.DataFrame(example_metrics)

# 自定义CSS样式
custom_css = """
.gr-box {border-radius: 10px; padding: 15px; background-color: #f9f9f9;}
.gr-dataframe table {width: 100%; border-collapse: collapse; max-height: 400px; overflow-y: auto; display: block;}
.gr-dataframe th {background-color: #4CAF50; color: white; padding: 8px; text-align: left; position: sticky; top: 0;}
.gr-dataframe td {padding: 8px; border-bottom: 1px solid #ddd;}
.gr-dataframe tr:nth-child(even) {background-color: #f2f2f2;}
"""

# 创建Gradio界面
with gr.Blocks(title="视频合并工具", css=custom_css) as interface:
    gr.Markdown("# 视频合并工具")
    gr.Markdown("上传两个视频，将它们水平合并成一个视频，并展示详细处理指标")

    with gr.Row():
        with gr.Column():
            video1 = gr.Video(label="视频 1")
            video2 = gr.Video(label="视频 2")
            submit_btn = gr.Button("合并视频", variant="primary")

        with gr.Column():
            output_video = gr.Video(label="合并后的视频")
            metrics_display = gr.JSON(label="实际处理指标")

    # 美化后的样例指标展示
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ### 📊 样例指标展示
                以下是一个典型视频合并任务的指标示例，帮助您了解输出结果的参考值。
                """
            )
            gr.Dataframe(
                value=example_df,
                headers=["指标名称", "值", "说明"],
                datatype=["str", "str", "str"],
                label="样例指标"
            )

    submit_btn.click(
        fn=merge_videos,
        inputs=[video1, video2],
        outputs=[output_video, metrics_display]
    )

# 启动界面
interface.launch(debug=True)