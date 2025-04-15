import os
import re
import cv2
import time
import warnings
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm
from typing import Union, List
from ultralytics import YOLO
from A25.UTIL.colorful import *
from A25.siri_utils.sleeper import Sleeper
from A25.global_config import GlobalConfig as cfg
from A25.pre.extract_number import extract_number
from A25.pre.dataloader import MemVidGanDataset, MemDEYOLO_Dataset
from torch.utils.data import DataLoader
from A25.siri_utils.preprocess import combime_wl_ir
from A25.global_config import GlobalConfig as cfg

original_filters = warnings.filters[:]
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    model_path = './A25/yolo_models/deyolo_medium/weights/best.pt'
    model = YOLO(model_path)
    tracker = sv.ByteTrack()
finally:
   warnings.filters = original_filters

def _predict(frame_or_batch):
    batch = frame_or_batch

    results = model.predict(
        batch,
        cfg=f"{cfg.root_dir}/yolo_model/game.yaml",
        imgsz=tuple(reversed(cfg.sz_wh)),
        # imgsz=640,
        stream=True,   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stream set
        conf=cfg.conf_threshold,
        iou=0.5,
        device=cfg.device,
        half=cfg.half,
        max_det=20,
        agnostic_nms=False,
        augment=False,
        vid_stride=False,
        visualize=False,
        verbose=False,

        # show_boxes=False,
        # show_labels=False,
        # show_conf=False,
        save=False,
        show=False,
        # batch=1
    )

    return results

def get_deyolo_vid_path(wl_vid :str, ir_vid :str, vid_to_annotate):
    assert os.path.exists(vid_to_annotate), f"ERROR: 视频文件不存在: {vid_to_annotate}"
    vid_to_annotate = cv2.VideoCapture(vid_to_annotate)
    if not vid_to_annotate.isOpened():
        print红(f"ERROR: 视频打开失败: {vid_to_annotate}")
        assert False
    # deyolo检测
    output_video= f"{cfg.outputdir}/output_deyolo.mp4"

    dataset = MemVidGanDataset(wl_vid, ir_vid=ir_vid, transform=None)

    # first_frame = combime_wl_ir(*dataset[0])
    first_frame = vid_to_annotate.read()[1]
    if first_frame is None:
        print红("ERROR: 视频帧读取失败")
        assert False
    vid_to_annotate.set(cv2.CAP_PROP_POS_FRAMES, 0)
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    try:
        for i, (wl_frame, ir_frame) in enumerate(dataset):
            print绿(f"\r testing frame {i}: {wl_frame.shape}, {ir_frame.shape}", end='')
            frame_to_annotate = vid_to_annotate.read()[1]

            if frame_to_annotate is None:
                print红(f"ERROR: 视频帧读取失败: {i}")
                break

            results = _predict([wl_frame, ir_frame,])
            if isinstance(results, list):
                result = results[0]
            else:
                result = next(results)

            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            annotator = sv.BoxAnnotator()
            # wl_frame = annotator.annotate(scene=wl_frame, detections=detections)
            # ir_frame = annotator.annotate(scene=ir_frame, detections=detections)
            frame_to_annotate = annotator.annotate(scene=frame_to_annotate, detections=detections)

            video_writer.write(frame_to_annotate)
    finally:
        video_writer.release()       

    return os.path.abspath(output_video)


def test_deyolo_metics(images_dir, labels_dir):

    output_dir = cfg.outputdir
    output_video= f"{cfg.outputdir}/output_deyolo.mp4"
    
    images_dir = 'datasets/deyolo_test'
    labels_dir = 'datasets/deyolo_test_labels'

    class_names = ["person"] 
    iou_thresholds = [0.5, 0.75]  
    confidence_thresholds = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94]
    
    from A25.detection_metrics import DetectionMetrics
    metrics = DetectionMetrics(iou_thresholds=iou_thresholds, class_names=class_names)
    dataset = MemDEYOLO_Dataset(images_dir, transform=None)
    images = dataset.mem
    labels = sorted(os.listdir(labels_dir),key=extract_number)
    if len(labels) != len(images):
        print红("Warning: len(labels) != len(images)")
        assert len(labels) > len(images)
        labels = labels[0:len(images)]
    assert len(labels) == len(images)
    N_images = len(images)

    
    first_frame = images[0][0]
    height, width = first_frame.shape[:2]
    

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    try: 
        for i in (range(N_images)):
            wl_frame = images[i][0]
            ir_frame = images[i][1]
            
            gt_file = os.path.join(labels_dir, labels[i])
            if not os.path.exists(gt_file):
                print(f"警告: 真实标签文件 {gt_file} 不存在")
                continue
            
            # 开始计时
            start_time = time.time()

            o = _predict([wl_frame, ir_frame,])
            if isinstance(o, list):
                result = o[0]
            else:
                result = next(o)

            
            # 结束计时
            inference_time = time.time() - start_time
            inference_time = inference_time*1000
            print(f"\r帧 {i}: 推理耗时 = {inference_time:.2f} ms", end='')

            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            annotator = sv.BoxAnnotator()
            wl_frame = annotator.annotate(scene=wl_frame, detections=detections)
            ir_frame = annotator.annotate(scene=ir_frame, detections=detections)

            video_writer.write(wl_frame)
            realtime_frame = f"{cfg.outputdir}/test_deyolo_metrics.jpg"
            cv2.imwrite(realtime_frame, wl_frame)

            yield realtime_frame, f"{inference_time:.2f} ms", f"{1/(inference_time/1000):.2f}", None, None, None
            
            frame_metrics = metrics.evaluate_image(gt_file, result, width, height, frame_idx=i)
            print("测试时间步长", time.time() - start_time)
        
    finally:
        video_writer.release()       
        results = metrics.compute_metrics(confidence_thresholds=confidence_thresholds)
        metrics.print_results(results)
    
    # 绘制PR曲线
    pr_curve = 'pr_curves.png'
    pr_curve = os.path.join(output_dir, pr_curve)
    metrics.plot_precision_recall_curves(save_path=os.path.join(output_dir, pr_curve))
    # 保存不同置信度阈值下的PR值到CSV文件
    # 为每个类别创建PR曲线数据
    assert metrics.num_classes == 1
    for class_id in range(metrics.num_classes):
        class_name = metrics.class_names[class_id]
        class_data = results['per_class'][class_name]
        output_csv = os.path.join(output_dir, f'{class_name}_pr_curve.csv')
        
        if 'thresholds' in class_data:
            # 准备数据
            conf_values = []
            precision_values = []
            recall_values = []
            
            for conf, pr_data in class_data['thresholds'].items():
                conf_values.append(conf)
                precision_values.append(pr_data['precision'])
                recall_values.append(pr_data['recall'])
            
            # 创建DataFrame
            pr_df = pd.DataFrame({
                'confidence': conf_values,
                'precision': precision_values,
                'recall': recall_values
            })
            
            # 保存到CSV
            pr_df.to_csv(output_csv, index=False)
    
    from A25.siri_utils.logger import print_dict
    print_dict(results)
    results.pop('per_class')
    
    yield cv2.cvtColor(wl_frame, cv2.COLOR_BGR2RGB), f"{inference_time:.2f} ms", f"{1/(inference_time/1000):.2f}", results, output_csv, pr_curve
