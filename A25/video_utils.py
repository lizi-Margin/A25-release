import cv2
from gradio import processing_utils

def _ensure_vid_format(output_video):
    if (processing_utils.ffmpeg_installed() and not processing_utils.video_is_playable(output_video)):
        # make sure no warnings from gradio
        output_video = processing_utils.convert_video_to_playable_mp4(output_video)
    return output_video


def get_wh(output_video):
    cap = cv2.VideoCapture(output_video)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width, _ = frame.shape
        else:
            print("无法读取第一帧。")
        cap.release()
    else:
        print("无法打开视频文件。")
    return (width, height)