import cv2
import numpy as np

def get_shaped_vid_path(near_ir_video_path: str, far_ir_video_path: str):

    # # 输入视频路径
    # near_ir_video_path = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/output_rgb_smoked3.mp4"  # 替换为你的近红外视频文件路径
    # far_ir_video_path = "C:/Users/xiwei/Desktop/服创/无锡低温烟雾环境双光视频/speed_output_tr_smoked3_25fps.mp4"    # 替换为你的远红外视频文件路径

    # 输出视频路径（可选）
    near_ir_output_path = "./output_get_shaped_vid_path_NIR.mp4"
    far_ir_output_path = "./output_get_shaped_vid_path_FIR.mp4"

    # 打开视频文件
    near_ir_cap = cv2.VideoCapture(near_ir_video_path)
    far_ir_cap = cv2.VideoCapture(far_ir_video_path)

    # 检查视频是否成功打开
    if not near_ir_cap.isOpened() or not far_ir_cap.isOpened():
        print("Error: Could not open one or both video files.")
        exit()

    # 获取视频的基本信息（以近红外视频为例）
    fps = near_ir_cap.get(cv2.CAP_PROP_FPS)  # 帧率
    # frame_width = 640  # 裁剪后的宽度 (672 - 32)
    # frame_height = 419  # 调整后的高度

    frame_width = 734  # 裁剪后的宽度 (672 - 32)
    frame_height = 480  # 调整后的高度

    # 定义视频写入器（可选，如果你想保存处理后的视频）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    near_ir_out = cv2.VideoWriter(near_ir_output_path, fourcc, fps, (frame_width, frame_height))
    far_ir_out = cv2.VideoWriter(far_ir_output_path, fourcc, fps, (frame_width, frame_height))

    # 逐帧处理视频
    while near_ir_cap.isOpened() and far_ir_cap.isOpened():
        ret1, frame_1 = near_ir_cap.read()
        ret2, frame_2 = far_ir_cap.read()
        if (not ret1) or (not ret2):
            print("End of video1/2 reached.")
            break
        rgb_resized = cv2.resize(frame_1, (704, 419), interpolation=cv2.INTER_AREA)
        rgb_cropped = rgb_resized[:, 32:672]

        ir_cropped = frame_2[66:485, :]

        rgb_resized = cv2.resize(rgb_cropped, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        ir_resized = cv2.resize(ir_cropped, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

        near_ir_out.write(rgb_resized)
        far_ir_out.write(ir_resized)


    # 释放资源
    near_ir_cap.release()
    far_ir_cap.release()
    near_ir_out.release()
    far_ir_out.release()
    cv2.destroyAllWindows()

    print("Video processing completed.")

    return near_ir_output_path, far_ir_output_path




def get_shaped_vid_path_(near_ir_video_path: str, far_ir_video_path: str):

    # 输出视频路径（可选）
    near_ir_output_path = "./output_get_shaped_vid_path_NIR.mp4"
    far_ir_output_path = "./output_get_shaped_vid_path_FIR.mp4"

    # 打开视频文件
    near_ir_cap = cv2.VideoCapture(near_ir_video_path)
    far_ir_cap = cv2.VideoCapture(far_ir_video_path)

    # 检查视频是否成功打开
    if not near_ir_cap.isOpened() or not far_ir_cap.isOpened():
        print("Error: Could not open one or both video files.")
        exit()

    # 获取视频的基本信息（以近红外视频为例）
    fps = near_ir_cap.get(cv2.CAP_PROP_FPS)  # 帧率
    # frame_width = 640  # 裁剪后的宽度 (672 - 32)
    # frame_height = 419  # 调整后的高度

    frame_width = 734  # 裁剪后的宽度 (672 - 32)
    frame_height = 480  # 调整后的高度

    # 定义视频写入器（可选，如果你想保存处理后的视频）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    near_ir_out = cv2.VideoWriter(near_ir_output_path, fourcc, fps, (frame_width, frame_height))
    far_ir_out = cv2.VideoWriter(far_ir_output_path, fourcc, fps, (frame_width, frame_height))

    # 逐帧处理视频
    while near_ir_cap.isOpened() and far_ir_cap.isOpened():
        ret1, frame_1 = near_ir_cap.read()
        ret2, frame_2 = far_ir_cap.read()
        if (not ret1) and (not ret2):
            break

        if ret1:
            rgb_resized = cv2.resize(frame_1, (704, 419), interpolation=cv2.INTER_AREA)
            rgb_cropped = rgb_resized[:, 32:672]
            rgb_resized = cv2.resize(rgb_cropped, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            near_ir_out.write(rgb_resized)

        if ret2:
            ir_cropped = frame_2[66:485, :]
            ir_resized = cv2.resize(ir_cropped, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            far_ir_out.write(ir_resized)


    # 释放资源
    near_ir_cap.release()
    far_ir_cap.release()
    near_ir_out.release()
    far_ir_out.release()
    cv2.destroyAllWindows()

    print("Video processing completed.")

    return near_ir_output_path, far_ir_output_path