import cv2
import torch
import numpy as np
from PIL import Image
import supervision as sv

from A25.global_config import GlobalConfig as cfg
import A25.video_fusion as video_fusion
import A25.dehaze as dehaze
import A25.deyolo as deyolo
import A25.yolo as yolo

def fuse_frame(wl_frame, ir_frame):
    device = cfg.device

    ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2RGB)
    wl_frame = cv2.cvtColor(wl_frame, cv2.COLOR_BGR2RGB)

    # 调整大小以匹配模型输入要求（如果需要）
    # 这里假设模型接受任意大小的输入，如果不是，需要调整大小

    # 转换为张量
    ir_tensor = torch.from_numpy(ir_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    wl_tensor = torch.from_numpy(wl_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 转移到设备并转换为半精度
    ir_tensor = ir_tensor.to(device).half()
    wl_tensor = wl_tensor.to(device).half()



    # 运行模型
    with torch.no_grad():
        # YCrCb转换
        vi_Y, vi_Cb, vi_Cr = video_fusion.RGB2YCrCb(wl_tensor)
        vi_Y = vi_Y.to(device).half()
        vi_Cb = vi_Cb.to(device).half()
        vi_Cr = vi_Cr.to(device).half()

        Seg_pred, _, _, fused_img, _, _ = video_fusion.MPF_model(wl_tensor, ir_tensor)

        # 转换回RGB
        fused_img = video_fusion.YCbCr2RGB(fused_img, vi_Cb, vi_Cr)

    # 转换为numpy数组以保存到视频
    fused_np = (fused_img[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # 转换为BGR（OpenCV格式）
    fused_np = cv2.cvtColor(fused_np, cv2.COLOR_RGB2BGR)
    return fused_np

def dehaze_frame(frame):
    device = cfg.device

    h, w = frame.shape[0:2]

    frame = Image.fromarray(frame)
    transform = dehaze.transform
    frame = transform(frame)
    assert len(frame.shape) == 3 and isinstance(frame, torch.Tensor)
    frame = frame.unsqueeze(0).to(device)

    color_images = dehaze.wl_to_color(frame)
    
    fake_images = dehaze.generator(color_images)
    fake_images = dehaze._post_compute(fake_images)[0]
    fake_images = cv2.resize(fake_images, (w, h))

    return cv2.cvtColor(fake_images, cv2.COLOR_RGB2BGR)

def deyolo_detect_frame(wl_frame, ir_frame):
    results = deyolo._predict([wl_frame, ir_frame,])
    if isinstance(results, list):
        result = results[0]
    else:
        result = next(results)

    detections = sv.Detections.from_ultralytics(result)
    detections = deyolo.tracker.update_with_detections(detections)

    annotator = sv.BoxAnnotator()
    wl_frame = annotator.annotate(scene=wl_frame, detections=detections)
    ir_frame = annotator.annotate(scene=ir_frame, detections=detections)
    return wl_frame, detections

def yolo_detect_frame(frame):
    results = yolo._predict(frame)
    if isinstance(results, list):
        result = results[0]
    else:
        result = next(results)

    detections = sv.Detections.from_ultralytics(result)
    detections = deyolo.tracker.update_with_detections(detections)

    annotator = sv.BoxAnnotator()
    frame = annotator.annotate(scene=frame, detections=detections)
    return frame, detections

def process_frame(wl_frame, ir_frame, use_deyolo):
    fused_frame = fuse_frame(wl_frame, ir_frame)
    
    dehazed_wl_frame = dehaze_frame(wl_frame)

    dehazed_fused_frame = fuse_frame(dehazed_wl_frame, ir_frame)

    if use_deyolo:
        yolo_frame, yolo_detections = deyolo_detect_frame(wl_frame, ir_frame)
    else:
        yolo_frame, yolo_detections = yolo_detect_frame(wl_frame)

    fused_frame_path = f"{cfg.outputdir}/fused_frame.jpg"
    cv2.imwrite(fused_frame_path, fused_frame)
    dehazed_fused_frame_path = f"{cfg.outputdir}/dehazed_fused_frame.jpg"
    cv2.imwrite(dehazed_fused_frame_path, dehazed_fused_frame)
    dehazed_wl_frame_path = f"{cfg.outputdir}/dehazed_wl_frame.jpg"
    cv2.imwrite(dehazed_wl_frame_path, dehazed_wl_frame)
    yolo_frame_path = f"{cfg.outputdir}/yolo_frame.jpg"
    cv2.imwrite(yolo_frame_path, yolo_frame)

    return {
        'fused_frame': fused_frame_path,
        'dehazed_fused_frame': dehazed_fused_frame_path,
        'dehazed_wl_frame': dehazed_wl_frame_path,
        'deyolo_frame': yolo_frame_path
    }

