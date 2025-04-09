import cv2, os
import torch
import warnings
import numpy as np
from tqdm import tqdm
from A25.third_party.PSFusion.utils import RGB2YCrCb, YCbCr2RGB
from A25.third_party.PSFusion.PSF import PSF
from A25.third_party.PSFusion.options import TestOptions 
from A25.third_party.PSFusion.saver import resume, save_img_single
from A25.global_config import GlobalConfig as cfg


original_filters = warnings.filters[:]
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    parser = TestOptions()
    opts = parser.parse()
    device = 'cuda'
    MPF_model = PSF(opts.class_nb).to(device)
    MPF_model = resume(MPF_model, model_save_path=opts.resume, device=device, is_train=False)
    MPF_model = MPF_model.half() 
    MPF_model.eval()
finally:
   warnings.filters = original_filters
    
def process_video(vi_video, ir_video):
    output_dir = cfg.outputdir
    output_video = os.path.join(output_dir, 'output_fusion.mp4')
    
    # 打开红外和可见光视频
    ir_cap = cv2.VideoCapture(ir_video)
    vi_cap = cv2.VideoCapture(vi_video)
    
    # 检查视频是否成功打开
    if not ir_cap.isOpened() or not vi_cap.isOpened():
        print("错误：无法打开视频文件")
        return ir_video
    
    # 获取视频参数
    width = int(vi_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vi_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width != int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH)):print(f"input width err")
    if height != int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):print(f"input height err")

    total_frames = int(min(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT), vi_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = vi_cap.get(cv2.CAP_PROP_FPS)  # 帧率
    assert fps == ir_cap.get(cv2.CAP_PROP_FPS), f"input fps err"  # 帧率
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 创建进度条
    progress_bar = tqdm(total=total_frames, desc="处理视频帧")
    
    with torch.no_grad():  # 内部操作不跟踪历史
        frame_idx = 0
        while True:
            # 读取红外和可见光帧
            ir_ret, ir_frame = ir_cap.read()
            vi_ret, vi_frame = vi_cap.read()
            # 检查是否到达视频末尾
            if not ir_ret or not vi_ret:
                break
            
            if ir_frame.shape != vi_frame.shape: ir_frame = cv2.resize(ir_frame, (width, height))
                
            # 预处理帧
            ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2RGB)
            vi_frame = cv2.cvtColor(vi_frame, cv2.COLOR_BGR2RGB)
            
            # 调整大小以匹配模型输入要求（如果需要）
            # 这里假设模型接受任意大小的输入，如果不是，需要调整大小
            
            # 转换为张量
            ir_tensor = torch.from_numpy(ir_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            vi_tensor = torch.from_numpy(vi_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # 转移到设备并转换为半精度
            ir_tensor = ir_tensor.to(device).half()
            vi_tensor = vi_tensor.to(device).half()
            
            # YCrCb转换
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(vi_tensor)
            vi_Y = vi_Y.to(device).half()
            vi_Cb = vi_Cb.to(device).half()
            vi_Cr = vi_Cr.to(device).half()
            
            # 运行模型
            Seg_pred, _, _, fused_img, _, _ = MPF_model(vi_tensor, ir_tensor)
            
            # 转换回RGB
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            
            # 转换为numpy数组以保存到视频
            fused_np = (fused_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # 转换为BGR（OpenCV格式）
            fused_np = cv2.cvtColor(fused_np, cv2.COLOR_RGB2BGR)
            
            # 写入输出视频
            out.write(fused_np)
            
            # 更新进度条
            progress_bar.update(1)
            frame_idx += 1
    
    # 释放资源
    progress_bar.close()
    ir_cap.release()
    vi_cap.release()
    out.release()
    print(f"融合视频已保存至 {output_video}")
    return output_video
