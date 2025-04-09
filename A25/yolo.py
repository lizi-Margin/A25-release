import os
import re
import sys
import cv2
import numpy as np
import supervision as sv
from typing import Union, List
from A25.UTIL.colorful import *
from A25.siri_utils.sleeper import Sleeper
from A25.global_config import GlobalConfig as cfg
from A25.pre.dataloader import MemVidGanDataset

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

model_path = './A25/yolo_models/best.pt'
tracker = sv.ByteTrack()

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
    output_video="./output_yolo.mp4"

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

