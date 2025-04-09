import cv2, torch, os
import numpy as np
import random
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from A25.pytorch_ssim import ssim
from A25.global_config import GlobalConfig as cfg


# def compute_similarity(frame1, frame2):
#     """计算两帧的结构相似性（SSIM）"""
#     return ssim(frame1, frame2)


def compute_similarity(frame1, frame2):
    frame1_tensor = torch.tensor(frame1, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    frame2_tensor = torch.tensor(frame2, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    if torch.cuda.is_available():
        frame1_tensor = frame1_tensor.cuda()
        frame2_tensor = frame2_tensor.cuda()

    return ssim(frame1_tensor, frame2_tensor).item()



def find_alignment_anchor(video_path1, video_path2, sample_size=20, window_size=100, output=None):
    """随机选择 sample_size 帧，找到最相似的一对帧作为锚点"""
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    frames1, frames2 = [], []
    
    while cap1.isOpened():
        ret1, frame1 = cap1.read()
        if not ret1:
            break
        frames1.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY))
    
    while cap2.isOpened():
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        frames2.append(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY))

    cap1.release()
    cap2.release()

    num_frames1, num_frames2 = len(frames1), len(frames2)

    # 从frames1中随机选择 sample_size 帧
    n = min(sample_size, num_frames1)
    # sampled_indices = random.sample(range(num_frames1), n)
    sampled_indices = random.sample(range(min(2 * sample_size, num_frames1)), n)

    best_match = -1
    best_frame1, best_frame2 = -1, -1

    # 计算随机选择的帧与所有帧的相似度
    for i in tqdm(sampled_indices, desc="查找锚点帧"):
        start_index = max(0, i - window_size)
        end_index = min(num_frames2, i + window_size)

        for j in tqdm(range(start_index, end_index), desc=f"对比 {i} 帧", leave=False):
            frame2_resized = cv2.resize(frames2[j], (frames1[i].shape[1], frames1[i].shape[0]))
            similarity = compute_similarity(frames1[i], frame2_resized)

            if similarity > best_match:
                best_match = similarity
                best_frame1, best_frame2 = i, j
    
    if output is not None:
        best_img1, best_img2 = frames1[best_frame1], frames2[best_frame2]

        # 保存最佳匹配帧
        best_img2_resized = cv2.resize(best_img2, (best_img1.shape[1], best_img1.shape[0]))
        best_merged_frame = np.hstack([best_img1, best_img2_resized])
        best_merged_frame = cv2.cvtColor(best_merged_frame, cv2.COLOR_GRAY2BGR)

        best_frame_path = f"{output}.best_frame.png"
        cv2.imwrite(best_frame_path, best_merged_frame)
        print(f"最佳匹配帧已保存至 {best_frame_path}")

    return best_frame1, best_frame2


def align_and_save_video(video_path1, video_path2, sample_size=20):
    output_dir = cfg.outputdir
    """基于锚点帧对齐视频"""
    best_frame1_idx, best_frame2_idx = find_alignment_anchor(video_path1, video_path2, sample_size, output=os.path.join(output_dir, 'output_time_alignment') if output_dir else './output_time_alignment')



    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 使用output_dir参数决定输出路径
    output_path1 = os.path.join(output_dir, 'output_time_alignment_video1.mp4') if output_dir else './output_time_alignment_video1.mp4'
    output_path2 = os.path.join(output_dir, 'output_time_alignment_video2.mp4') if output_dir else './output_time_alignment_video2.mp4'

    fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter(output_path1, fourcc1, fps, (width, height))
    fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
    out2 = cv2.VideoWriter(output_path2, fourcc2, fps, (width, height))

    frame_index = 0

    # 计算起始帧偏移量
    offset = best_frame2_idx - best_frame1_idx
    if offset > 0:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, offset)
    else:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, -offset)

    for frame_index in tqdm(range(int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))), desc="生成对齐视频"):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.resize(frame2_gray, (width, height))

        frame1 = cv2.cvtColor(frame1_gray, cv2.COLOR_GRAY2BGR)
        frame2 = cv2.cvtColor(frame2_gray, cv2.COLOR_GRAY2BGR)
        # merged_frame = np.hstack([frame1_gray, frame2_gray])
        
        out1.write(frame1)
        out2.write(frame2)

        # # 实时显示对齐情况
        # cv2.imshow("Aligned Frames", merged_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap1.release()
    cap2.release()
    out1.release()
    out2.release()
    # cv2.destroyAllWindows()
    return output_path1, output_path2

    
def get_aligned_vid_path(video1, video2):
    return align_and_save_video(video1, video2, sample_size=20)


def test_alignment(video1, video2):
    for i in range(100):
        find_alignment_anchor(video1, video2, 20, output=f"{cfg.outputdir}/alignment_test_{i}")

if __name__ == "__main__":
    video1 = "rgb_smoked1-shaped.mp4"
    video2 = "tr_smoked1-shaped.mp4"
    test_alignment(video1, video2)
