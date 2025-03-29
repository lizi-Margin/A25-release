import os, cv2, torch
from tqdm import tqdm
from A25.UTIL.colorful import *
from gradio import processing_utils

from A25.dehaze import get_dehazed_vid_path
from A25.deyolo import get_deyolo_vid_path
from A25.video_fusion import process_video
from A25.stream_process import process_frame

def _ensure_vid_format(output_video):
    if (processing_utils.ffmpeg_installed() and not processing_utils.video_is_playable(output_video)):
        # make sure no warnings from gradio
        output_video = processing_utils.convert_video_to_playable_mp4(output_video)
    return output_video

def print_vid_info(video):
    print(f"type={type(video)}, value={repr(video)}")

def run_model(wl_video, ir_video):
    print蓝("[run_model] called")
    assert isinstance(wl_video, str) and isinstance(ir_video, str)
    print_vid_info(wl_video)
    print_vid_info(ir_video)

    if (not os.path.isfile(wl_video)) or (not os.path.isfile(ir_video)):
        print红(f"[run_model] Error: input video file not found")
        print红(f"[run_model] return value = (None, None)")
        return None, None

    dehazed_o, deyolo_o = get_dehazed_vid_path(wl_video), get_deyolo_vid_path(wl_video, ir_video)
    dehazed_o, deyolo_o = _ensure_vid_format(dehazed_o), _ensure_vid_format(deyolo_o)

    print蓝(f"[run_model] return value = ({dehazed_o}, {deyolo_o})")
    return dehazed_o, deyolo_o

def run_fusion(wl_video, ir_video):
    print蓝("[run_fusion] called")
    assert isinstance(wl_video, str) and isinstance(ir_video, str)
    print_vid_info(wl_video)
    print_vid_info(ir_video)

    if (not os.path.isfile(wl_video)) or (not os.path.isfile(ir_video)):
        print红(f"[run_fusion] Error: input video file not found")
        print红(f"[run_fusion] return value = (None,)")
        return None

    fusion_o = process_video(wl_video, ir_video)
    fusion_o = _ensure_vid_format(fusion_o)

    print蓝(f"[run_model] return value = ({fusion_o},)")
    return fusion_o

def get_stream_iter(aligned_wl_video, aligned_ir_video):
    wl_cap = cv2.VideoCapture(aligned_wl_video)
    ir_cap = cv2.VideoCapture(aligned_ir_video)
    
    if not ir_cap.isOpened() or not wl_cap.isOpened():
        print("错误：无法打开视频文件")
        return
    
    width = int(wl_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(wl_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert width == int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    assert height == int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(min(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT), wl_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    fps = wl_cap.get(cv2.CAP_PROP_FPS)  # 帧率
    assert fps == ir_cap.get(cv2.CAP_PROP_FPS)  # 帧率
    
    progress_bar = tqdm(total=total_frames, desc="处理视频帧")
    
    with torch.no_grad():
        frame_idx = 0
        while True:
            wl_ret, wl_frame = wl_cap.read()
            ir_ret, ir_frame = ir_cap.read()
            if not ir_ret or not wl_ret: break
            
            o = process_frame(wl_frame, ir_frame)
            
            yield o['fused_frame'], o['dehazed_fused_frame'], o['dehazed_wl_frame'], o['deyolo_frame']

            progress_bar.update(1)
            frame_idx += 1
