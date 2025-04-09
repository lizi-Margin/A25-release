import gradio as gr
import cv2
import numpy as np
import os
import time
from sklearn.metrics import precision_score, recall_score
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from A25.UTIL.colorful import *
from A25 import run_model, run_fusion, get_aligned_vid_path, get_stream_iter

def has_None(*args):
    for x in args:
        if x is None:
            return True
    return False

def run_batch(wl, ir, fusion_method):
    feature_level_fusion = (fusion_method == "特征融合")
    if has_None(wl, ir):
        print红("ERROR: got None in args")
        return

    aligned_wl, aligned_ir = get_aligned_vid_path(wl, ir) 
    yield aligned_wl, aligned_ir, None, None, None 
    print绿(f"对齐完成: {aligned_wl}, {aligned_ir}")

    fusion_o = run_fusion(aligned_wl, aligned_ir)
    yield aligned_wl, aligned_ir, fusion_o, None, None 
    print绿(f"融合完成: {fusion_o}")

    dehazed_o, deyolo_o = run_model(aligned_wl, aligned_ir)
    yield aligned_wl, aligned_ir, fusion_o, dehazed_o, deyolo_o
    print绿(f"去雾/检测完成: {dehazed_o}, {deyolo_o}")

def run_stream(wl, ir, fusion_method):
    feature_level_fusion = (fusion_method == "特征融合")
    if has_None((wl, ir,)): 
        print红("ERROR: got None in args")
        return
    aligned_wl, aligned_ir = get_aligned_vid_path(wl, ir) 

    result = get_stream_iter(aligned_wl, aligned_ir, use_deyolo=feature_level_fusion)
    if result is None:
        print红("ERROR: stream generator is None")
        return
    
    try:
        while True:
            res = next(result)
            ret = [aligned_wl, aligned_ir]
            for x in res:
                ret.append(x)
            yield ret
    except StopIteration:
        pass

def run_detect_metrics(imagedir_path, label_path, fusion_method):
    feature_level_fusion = (fusion_method == "特征融合")
    if feature_level_fusion:
        from A25.deyolo import test_deyolo_metics
        result =  test_deyolo_metics(imagedir_path, label_path)
        try:
            while True:
                res = next(result)
                yield res
        except StopIteration:
            pass
        
    else:
        from A25.yolo import test_yolo_metics
        result = test_yolo_metics(imagedir_path, label_path)
        try:
            while True:
                res = next(result)
                yield res
        except StopIteration:
            pass




# 自定义CSS样式
custom_css = """
.gr-box {border-radius: 10px; padding: 15px; background-color: #f9f9f9;}
.gr-dataframe table {width: 100%; border-collapse: collapse; max-height: 400px; overflow-y: auto; display: block;}
.gr-dataframe th {background-color: #4CAF50; color: white; padding: 8px; text-align: left; position: sticky; top: 0;}
.gr-dataframe td {padding: 8px; border-bottom: 1px solid #ddd;}
.gr-dataframe tr:nth-child(even) {background-color: #f2f2f2;}
"""

with gr.Blocks(title="A25-release", css=custom_css) as interface:
    fusion_method = gr.Radio(
        choices=["特征融合", "视频融合"],
        value="特征融合",
        label="融合方法选择",
        info="选择使用的融合方法",
        interactive=True
    )

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
        inputs=[wl_video, ir_video, fusion_method],
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
            run_stream_btn = gr.Button("RUN STREAM", variant="primary")

        with gr.Column():
            # aligned_wl_video_stream_input = gr.Video(label="近红外对齐视频")
            # aligned_ir_video_stream_input = gr.Video(label="热成像对齐视频")
            fused_frame = gr.Image(label="双通道融合结果fused_frame")
            dehazed_fused_frame = gr.Image(label="双通道融合去烟结果dehazed_fused_frame")
            dehazed_wl_frame = gr.Image(label="近红外去烟结果dehazed_wl_frame")
            deyolo_frame = gr.Image(label="识别结果")
    
    run_stream_btn.click(
        fn=run_stream,
        inputs=[wl_video, ir_video, fusion_method],
        outputs=[aligned_wl_vid, aligned_ir_vid, fused_frame, dehazed_fused_frame, dehazed_wl_frame, deyolo_frame],
        show_progress=True
    )


    gr.Markdown(
        """
        # 检测性能测试
        """
    )
    with gr.Row():
        with gr.Column():
            run_detect_metrics_btn = gr.Button("RUN DETECT METRICS", variant="primary")
            imagedir_path = gr.Textbox(label="图片目录", placeholder="输入图片目录路径")
            label_path = gr.Textbox(label="标签文件", placeholder="输入标签文件路径")
            
            realtime_detect_output = gr.Image(label="实时检测结果")
            realtime_detect_delay = gr.Textbox(label="实时检测延迟", placeholder="实时检测延迟")
            realtime_detect_fps = gr.Textbox(label="实时检测FPS", placeholder="实时检测FPS")
        
        with gr.Column():
            performance_metrics = gr.JSON(label="性能指标")
            # performance_metrics = gr.Dataframe(label="性能指标", type="pandas", col_count=2, headers=['metric', 'value'])
            pr_curve = gr.Image(label="PR曲线")
            pr_table = gr.Dataframe(label="PR表格", type="pandas", col_count=3, headers=['confidence', 'precision', 'recall'])
            
            
            
            
        run_detect_metrics_btn.click(
            fn=run_detect_metrics,
            inputs=[imagedir_path, label_path, fusion_method],
            outputs=[realtime_detect_output, realtime_detect_delay, realtime_detect_fps, performance_metrics, pr_table, pr_curve]
        )


# 启动界面
interface.launch(debug=True, pwa=False)