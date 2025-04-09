import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from A25.global_config import GlobalConfig as cfg
from A25.UTIL.colorful import *
from A25.siri_utils.preprocess import combime_wl_ir
from A25.wl_to_color import wl_to_color



from A25.pre.dataloader import GanDataset, MemGanDataset, MemVidGanDataset
# from net.FFA import FFA
from A25.third_party.DWGAN.model import fusion_net
from A25.pre.transform import transform_dwgan as transform
from A25.net.model_io import load_gan_model
from A25.siri_utils.preprocess import _post_compute

def cat(input_frame, real_frame, fake_frame):
    return combime_wl_ir(combime_wl_ir(input_frame, real_frame), fake_frame)



device = cfg.device
generator = fusion_net().to(device)
model_path = "./A25/third_party/DWGAN/weights/dw_gan_best.pkl"
generator.load_state_dict(torch.load(model_path, weights_only=True))

generator.eval()
    

def get_dehazed_vid_path(wl_video: str, output_video=None) -> str:
    if output_video is None: output_video = "./output_deahze.mp4"
    batch_size = 48
    fps = 25
    
    dataset = MemVidGanDataset(wl_vid=wl_video, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    first_batch_input, first_batch_real = next(iter(dataloader))
    first_input = _post_compute(first_batch_input)[0]
    first_real = _post_compute(first_batch_real)[0]
    # with torch.no_grad(): first_fake = _post_compute(generator(first_batch_input.to(device)))[0]
    first_fake = first_real
    height, width, _ = cat(first_input, first_real, first_fake).shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    try:
        with torch.no_grad():
            for k, (wl_images, _) in enumerate(dataloader):
                wl_images = wl_images.to(device)
                color_images = wl_to_color(wl_images)
                
                fake_images = generator(color_images)
                fake_images = _post_compute(fake_images)
                wl_images = _post_compute(wl_images)
                color_images = _post_compute(color_images)



                for i in range(fake_images.shape[0]):
                    wl_frame = cv2.cvtColor(wl_images[i], cv2.COLOR_RGB2BGR)
                    color_frame = cv2.cvtColor(color_images[i], cv2.COLOR_RGB2BGR)
                    fake_frame = cv2.cvtColor(fake_images[i], cv2.COLOR_RGB2BGR)
                    video_writer.write(cat(wl_frame, color_frame, fake_frame))
                    print绿(f"\rProcessing batch: {k}", end='')
    finally:
        video_writer.release()
        print绿(f"Video saved at {output_video}")
    
    return os.path.abspath(output_video)


def get_dehazed_vid_path_(wl_video: str, output_video=None) -> str:
    if output_video is None: output_video = "./output_deahze.mp4"
    batch_size = 48
    fps = 25
    

    dataset = MemVidGanDataset(wl_vid=wl_video, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    tgt_size = (734, 480,)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, tgt_size)

    try:
        with torch.no_grad():
            for k, (wl_images, _) in enumerate(dataloader):
                wl_images = wl_images.to(device)
                color_images = wl_to_color(wl_images)
                
                fake_images = generator(color_images)
                fake_images = _post_compute(fake_images)
                wl_images = _post_compute(wl_images)
                color_images = _post_compute(color_images)



                for i in range(fake_images.shape[0]):
                    wl_frame = cv2.cvtColor(wl_images[i], cv2.COLOR_RGB2BGR)
                    color_frame = cv2.cvtColor(color_images[i], cv2.COLOR_RGB2BGR)
                    fake_frame = cv2.cvtColor(fake_images[i], cv2.COLOR_RGB2BGR)
                    video_writer.write(bw_to_rgb(rgb_to_bw(cv2.resize(fake_frame, tgt_size))))
                    print绿(f"\rProcessing batch: {k}", end='')
    finally:
        video_writer.release()
        print绿(f"Video saved at {output_video}")
    
    return os.path.abspath(output_video)

def rgb_to_bw(frame):
    """
    Convert an RGB frame (np.ndarray) to a black - white frame.
    """
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        bw_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return bw_frame
    else:
        print("Input is not a valid RGB frame.")
        raise ValueError


def bw_to_rgb(frame):
    """
    Convert a black - white frame (np.ndarray) to an RGB frame.
    """
    if len(frame.shape) == 2:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return rgb_frame
    else:
        print("Input is not a valid black - white frame.")
        raise ValueError
