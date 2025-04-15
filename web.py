import gradio as gr
import sys
import cv2
import numpy as np
import os
import time
from sklearn.metrics import precision_score, recall_score
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from A25.UTIL.colorful import *
from A25 import run_model, run_fusion, get_aligned_vid_path, get_stream_iter
from datetime import datetime
import argparse

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
    print绿(f"融合完成: {fusion_o}")

    dehazed_o, deyolo_o = run_model(aligned_wl, aligned_ir, vid_to_annotate=fusion_o)

    yield aligned_wl, aligned_ir, dehazed_o, fusion_o, deyolo_o
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

        if fusion_method == "直接识别": 
            imagedir_path = 'datasets/wl_test'
            result = test_yolo_metics(imagedir_path, label_path, imagedir_path)
        else:
            import A25.yolo as yolo
            yolo.model = yolo.fused_model
            imagedir_path = 'datasets/fused_test'
            result = test_yolo_metics(imagedir_path, label_path, "./datasets/dehazed_fused_test")
        try:
            while True:
                res = next(result)
                yield res
        except StopIteration:
            pass




# 自定义CSS样式
custom_css = """
/* 整体主题 */
:root {
    --primary-color: #2c3e50;
    --secondary-color: #34495e;
    --accent-color: #3498db;
    --background-color: #ecf0f1;
}

/* 容器样式 */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* 标题样式 */
h1, h2, h3 {
    color: var(--primary-color);
    border-bottom: 2px solid var(--accent-color);
    padding-bottom: 10px;
    margin-bottom: 20px;
}

/* 卡片样式 */
.gr-box {
    border-radius: 15px;
    padding: 20px;
    background-color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    transition: transform 0.2s;
}

.gr-box:hover {
    transform: translateY(-2px);
}

/* 按钮样式 */
.gr-button {
    background-color: var(--accent-color);
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    transition: all 0.3s;
}

.gr-button:hover {
    background-color: var(--primary-color);
    transform: scale(1.05);
}

/* 表格样式 */
.gr-dataframe {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.gr-dataframe table {
    width: 100%;
    border-collapse: collapse;
    max-height: 400px;
    overflow-y: auto;
    display: block;
}

.gr-dataframe th {
    background-color: var(--primary-color);
    color: white;
    padding: 12px;
    text-align: left;
    position: sticky;
    top: 0;
}

.gr-dataframe td {
    padding: 12px;
    border-bottom: 1px solid #ddd;
}

.gr-dataframe tr:nth-child(even) {
    background-color: var(--background-color);
}

"""

# 替换原有的 LogCapture 类和相关代码
class LogManager:
    def __init__(self, log_file):
        self.log_file = log_file
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # 清空或创建日志文件
        open(self.log_file, 'w').close()
    
    def get_logs(self, n_lines=None):
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                if n_lines:
                    return ''.join(lines[-n_lines:])
                return ''.join(lines)
        except Exception as e:
            return f"Error reading logs: {str(e)}"

# 替换更新日志的函数
def update_logs():
    while True:
        yield log_manager.get_logs(n_lines=1)
        time.sleep(0.2)

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='A25 Web Interface')
    parser.add_argument('--log', type=str, help='Log file path')
    return parser.parse_args()

with gr.Blocks(title="A25-release", css=custom_css) as interface:
    # 共用的融合方法选择
    with gr.Row():
        fusion_method = gr.Radio(
            choices=["特征融合", "图像融合", "直接识别"],
            value="特征融合",
            label="融合方法选择",
            info="选择使用的融合方法",
            interactive=True
        )
    
    # 使用选项卡分隔三个功能页面
    with gr.Tabs():
        # 批处理页面
        with gr.Tab("批处理测试"):
            with gr.Row():
                with gr.Column():
                    wl_video = gr.Video(label="近红外视频")
                    ir_video = gr.Video(label="热成像视频")
                    run_batch_btn = gr.Button("RUN Batch", variant="primary")

                with gr.Column():
                    aligned_wl_vid = gr.Video(label="对齐结果NIR", autoplay=True)
                    aligned_ir_vid = gr.Video(label="对齐结果FIR", autoplay=True)
                    dehazed_vid = gr.Video(label="近红外去烟视频", autoplay=True)
                    fusion_video = gr.Video(label="双通道融合结果", autoplay=True)
                    detected_vid = gr.Video(label="识别视频", autoplay=True)
            


        # 流处理页面
        with gr.Tab("流处理测试"):
            with gr.Row():
                with gr.Column():
                    wl_video_stream = gr.Video(label="近红外视频")
                    ir_video_stream = gr.Video(label="热成像视频")
                    run_stream_btn = gr.Button("RUN STREAM", variant="primary")

                with gr.Column():
                    aligned_wl_vid_stream = gr.Video(label="对齐结果NIR")
                    aligned_ir_vid_stream = gr.Video(label="对齐结果FIR")
                    dehazed_wl_frame = gr.Image(label="近红外去烟结果")
                    fused_frame = gr.Image(label="双通道融合结果")
                    deyolo_frame = gr.Image(label="识别结果")
            
            

        # 性能测试页面
        with gr.Tab("检测性能测试"):
            with gr.Row():
                with gr.Column():
                    run_detect_metrics_btn = gr.Button("RUN DETECT METRICS", variant="primary")
                    imagedir_path = gr.Textbox(label="图片目录", placeholder="输入图片目录路径", value='datasets/test/')
                    label_path = gr.Textbox(label="标签文件", placeholder="输入标签文件路径", value='datasets/deyolo_test_labels/')
                    
                    realtime_detect_output = gr.Image(label="实时检测结果")
                    realtime_detect_delay = gr.Textbox(label="实时检测延迟", placeholder="实时检测延迟")
                    realtime_detect_fps = gr.Textbox(label="实时检测FPS", placeholder="实时检测FPS")
                
                with gr.Column():
                    performance_metrics = gr.JSON(label="性能指标")
                    pr_curve = gr.Image(label="PR曲线")
                    pr_table = gr.Dataframe(
                        label="PR表格", 
                        type="pandas",
                        headers=['confidence', 'precision', 'recall']
                    )
                    
                

    # 修改日志显示区域
    with gr.Column():
        logs_output = gr.TextArea(
            label="运行日志",
            interactive=False,
            autoscroll=True,
            lines=1,
            max_lines=1,
            scale=5,
            show_label=False,
        )

    # # 每秒更新日志
    # logs_output.update(
    #     value=update_logs,
    #     every=1
    # )

    
    run_detect_metrics_btn.click(
        fn=run_detect_metrics,
        inputs=[imagedir_path, label_path, fusion_method],
        outputs=[realtime_detect_output, realtime_detect_delay, realtime_detect_fps, performance_metrics, pr_table, pr_curve]
    )
    
    run_stream_btn.click(
        fn=run_stream,
        inputs=[wl_video_stream, ir_video_stream, fusion_method],
        outputs=[aligned_wl_vid_stream, aligned_ir_vid_stream, dehazed_wl_frame, fused_frame, deyolo_frame],
        show_progress=True
    )

    run_batch_btn.click(
        fn=run_batch,
        inputs=[wl_video, ir_video, fusion_method],
        outputs=[aligned_wl_vid, aligned_ir_vid, dehazed_vid, fusion_video, detected_vid],
        show_progress=True
    )

    interface.load(
        fn=update_logs,
        outputs=logs_output,
    )

# 使用 tee 命令启动程序
if __name__ == "__main__":
    args = parse_args()
    log_file = args.log or f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_manager = LogManager(log_file)
    
    interface.launch(
        debug=True,
        pwa=False,
        quiet=True,
        server_name="0.0.0.0",
        server_port=7860
    )