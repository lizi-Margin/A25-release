import os
import re
import cv2
import warnings
import numpy as np
import supervision as sv
from typing import Union, List
from ultralytics import YOLO
from A25.UTIL.colorful import *
from A25.siri_utils.sleeper import Sleeper
from A25.global_config import GlobalConfig as cfg
from A25.pre.extract_number import extract_number
from A25.pre.dataloader import MemVidGanDataset
from torch.utils.data import DataLoader
from A25.siri_utils.preprocess import combime_wl_ir


original_filters = warnings.filters[:]
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    model_path = './A25/deyolo_models/best.pt'
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

def get_deyolo_vid_path(wl_vid :str, ir_vid :str):
    output_video="./output_deyolo.mp4"

    dataset = MemVidGanDataset(wl_vid, ir_vid=ir_vid, transform=None)

    first_frame = combime_wl_ir(*dataset[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    try:
        for i, (wl_frame, ir_frame) in enumerate(dataset):
            print绿(f"\r testing frame {i}: {wl_frame.shape}, {ir_frame.shape}", end='')

            results = _predict([wl_frame, ir_frame,])
            if isinstance(results, list):
                result = results[0]
            else:
                result = next(results)

            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            annotator = sv.BoxAnnotator()
            wl_frame = annotator.annotate(scene=wl_frame, detections=detections)
            ir_frame = annotator.annotate(scene=ir_frame, detections=detections)

            video_writer.write(combime_wl_ir(wl_frame, ir_frame))
    finally:
        video_writer.release()       

    return os.path.abspath(output_video)



