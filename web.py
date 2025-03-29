import gradio as gr
import cv2
import numpy as np
import os
import time
from sklearn.metrics import precision_score, recall_score
from skimage.metrics import structural_similarity as ssim
import pandas as pd

def run_batch(wl, ir):
    from align import get_aligned_vid_path
    from A25 import run_model, run_fusion
    from A25.UTIL.colorful import print绿

    aligned_wl, aligned_ir = get_aligned_vid_path(wl, ir) 
    yield aligned_wl, aligned_ir, None, None, None 
    print绿(f"对齐完成: {aligned_wl}, {aligned_ir}")

    fusion_o = run_fusion(aligned_wl, aligned_ir)
    yield aligned_wl, aligned_ir, fusion_o, None, None 
    print绿(f"融合完成: {fusion_o}")

    dehazed_o, deyolo_o = run_model(aligned_wl, aligned_ir)
    yield aligned_wl, aligned_ir, fusion_o, dehazed_o, deyolo_o
    print绿(f"去雾/检测完成: {dehazed_o}, {deyolo_o}")


# 自定义CSS样式
custom_css = """
.gr-box {border-radius: 10px; padding: 15px; background-color: #f9f9f9;}
.gr-dataframe table {width: 100%; border-collapse: collapse; max-height: 400px; overflow-y: auto; display: block;}
.gr-dataframe th {background-color: #4CAF50; color: white; padding: 8px; text-align: left; position: sticky; top: 0;}
.gr-dataframe td {padding: 8px; border-bottom: 1px solid #ddd;}
.gr-dataframe tr:nth-child(even) {background-color: #f2f2f2;}
"""

with gr.Blocks(title="A25-release", css=custom_css) as interface:
    gr.Markdown(
        """
        # 批处理测试
        """
    )
    with gr.Row():
        with gr.Column():
            wl_video = gr.Video(label="近红外视频")
            ir_video = gr.Video(label="热成像视频")
            run_batch_btn = gr.Button("RUN Batch", variant="primary")

        with gr.Column():
            aligned_wl_vid = gr.Video(label="对齐结果NIR")
            aligned_ir_vid = gr.Video(label="对齐结果FIR")
            fusion_video = gr.Video(label="双通道融合结果")
            dehazed_vid = gr.Video(label="去烟视频")
            detected_vid = gr.Video(label="识别视频")
    
    run_batch_btn.click(
        fn=run_batch,
        inputs=[wl_video, ir_video],
        outputs=[aligned_wl_vid, aligned_ir_vid, fusion_video, dehazed_vid, detected_vid],
        show_progress=True
    )

    gr.Markdown(
        """
        # 流处理测试
        """
    )
    with gr.Row():
        with gr.Column():
            aligned_wl_video_stream_input = gr.Video(label="近红外对齐视频")
            aligned_ir_video_stream_input = gr.Video(label="热成像对齐视频")
            run_stream_btn = gr.Button("RUN STREAM", variant="primary")

        with gr.Column():
            fused_frame = gr.Image(label="双通道融合结果fused_frame")
            dehazed_fused_frame = gr.Image(label="双通道融合去烟结果dehazed_fused_frame")
            dehazed_wl_frame = gr.Image(label="近红外去烟结果dehazed_wl_frame")
            deyolo_frame = gr.Image(label="deyolo识别结果")
    
    from A25 import get_stream_iter
    run_stream_btn.click(
        fn=get_stream_iter,
        inputs=[aligned_wl_video_stream_input, aligned_ir_video_stream_input],
        outputs=[fused_frame, dehazed_fused_frame, dehazed_wl_frame, deyolo_frame],
        show_progress=True
    )


# 启动界面
interface.launch(debug=True, pwa=False)