import os
import re
import sys
import cv2
import time
import numpy as np
import pandas as pd
import supervision as sv
from typing import Union, List
from A25.UTIL.colorful import *
from A25.siri_utils.sleeper import Sleeper
from A25.global_config import GlobalConfig as cfg
from A25.pre.dataloader import MemVidGanDataset, MemDEYOLO_Dataset
from A25.pre.extract_number import extract_number

def patch_module_namespace(base_name: str, alias_name: str):
    """将 sys.modules 中 base_name 及其子模块 替换为 alias_name 的对应模块。"""
    patched_modules = {}
    for name in list(sys.modules):
        if name == base_name or name.startswith(base_name + '.'):
            alias_sub_name = name.replace(base_name, alias_name, 1)
            if alias_sub_name in sys.modules:
                patched_modules[name] = sys.modules[name]
                sys.modules[name] = sys.modules[alias_sub_name]
    return patched_modules 

def restore_module_namespace(patched_modules: dict):
    """还原被 patch_module_namespace 替换的模块。"""
    for name, module in patched_modules.items():
        sys.modules[name] = module

def init_yolo_model(model_path):
    original_sys_modules = sys.modules.copy()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    import ultralytics
    import A25.yolo_ultralytics
    patched_modules = patch_module_namespace('ultralytics', 'A25.yolo_ultralytics')

    try:
        from A25.yolo_ultralytics import YOLO
        model = YOLO(model=model_path, task='detect')
    finally:
        restore_module_namespace(patched_modules)
        pass
    return model

tracker = sv.ByteTrack()
wl_model = init_yolo_model('./A25/yolo_models/wl/weights/best.pt')
fused_model = init_yolo_model('./A25/yolo_models/fused/weights/best.pt')
model = wl_model




def _predict(frame_or_batch: Union[np.ndarray, List[np.ndarray]]):
    if isinstance(frame_or_batch, np.ndarray):
        batch = [frame_or_batch]
    elif isinstance(frame_or_batch, list):
        assert isinstance(frame_or_batch[0], np.ndarray)
        batch = frame_or_batch
    else:
        assert False

    assert len(batch[0].shape) == 3
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    # frame = pad(frame, to_sz_wh=cfg.sz_wh)
    # frame = cv2.resize(frame, cfg.sz_wh)
    # if cfg.manual_preprocess: 
    #     batch = preprocess(batch)

    results = model.predict(
        batch,
        cfg=f"{cfg.root_dir}/A25/yolo_model/game.yaml",
        imgsz=tuple(reversed(cfg.sz_wh)),
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
        show_boxes=False,
        show_labels=False,
        show_conf=False,
        save=False,
        show=False,
        # batch=1
    )

    return results

def get_yolo_vid_path(wl_vid :str):
    output_video=f"{cfg.outputdir}/output_yolo.mp4"

    dataset = MemVidGanDataset(wl_vid, transform=None)

    first_frame = dataset[0][0]
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    try:
        for i, (frame, _) in enumerate(dataset):
            print绿(f"\r testing frame {i}: {frame.shape}", end='')
            assert len(frame.shape) == 3
            results = _predict(frame)
            if isinstance(results, list):
                result = results[0]
            else:
                result = next(results)

            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            annotator = sv.BoxAnnotator()
            frame = annotator.annotate(scene=frame, detections=detections)

            video_writer.write(frame)
    finally:
        video_writer.release()       

    return os.path.abspath(output_video)


def test_yolo_metics(images_dir, labels_dir):

    output_dir = cfg.outputdir
    output_video= f"{cfg.outputdir}/output_yolo.mp4"
    images_dir = 'datasets/yolo_test'
    labels_dir = 'datasets/yolo_test_labels'

    class_names = ["person"] 
    iou_thresholds = [0.5, 0.75]  
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
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

    
    first_frame = images[0]
    height, width = first_frame.shape[:2]
    

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    try: 
        for i in (range(N_images)):
            wl_frame = images[i]
            
            gt_file = os.path.join(labels_dir, labels[i])
            if not os.path.exists(gt_file):
                print(f"警告: 真实标签文件 {gt_file} 不存在")
                continue
            
            # 开始计时
            start_time = time.time()

            o = _predict(wl_frame)
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

            video_writer.write(wl_frame)
            realtime_frame = f"{cfg.outputdir}/test_deyolo_metrics.jpg"
            cv2.imwrite(realtime_frame, wl_frame)

            yield realtime_frame, f"{inference_time:.2f} ms", f"{1/(inference_time/1000):.2f}", None, None, None
            
            frame_metrics = metrics.evaluate_image(gt_file, result, width, height, frame_idx=i)
        
    finally:
        video_writer.release()       
        results = metrics.compute_metrics(confidence_thresholds=confidence_thresholds)
        metrics.print_results(results)
    
    # 绘制PR曲线
    pr_curve = 'pr_curves.png'
    pr_curve = os.path.join(output_dir, pr_curve)
    metrics.plot_precision_recall_curves(save_path=pr_curve)
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
    
    yield realtime_frame, f"{inference_time:.2f} ms", f"{1/(inference_time/1000):.2f}", results, output_csv, pr_curve

