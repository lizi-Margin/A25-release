import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch
import os
import cv2
import numpy as np
from A25.global_config import GlobalConfig as cfg
from A25.UTIL.colorful import *
from A25.siri_utils.preprocess import combime_wl_ir
from A25.wl_to_color import wl_to_color
from A25.video_utils import get_wh


from A25.pre.dataloader import GanDataset, MemGanDataset, MemVidGanDataset
# from net.FFA import FFA
from A25.third_party.DWGAN.model import fusion_net
from A25.net.model_io import load_gan_model
from A25.siri_utils.preprocess import _post_compute

def cat(input_frame, real_frame, fake_frame):
    return combime_wl_ir(combime_wl_ir(input_frame, real_frame), fake_frame)


device = cfg.device
generator = fusion_net().to(device)
model_path = "./A25/third_party/DWGAN/weights/dw_gan_best.pkl"
generator.load_state_dict(torch.load(model_path, weights_only=True))

generator.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
    

def get_dehazed_vid_path(wl_video: str) -> str:
    # 去烟
    batch_size = 12
    

    dataset = MemVidGanDataset(wl_vid=wl_video, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    output_video = os.path.join(cfg.outputdir, "output_deahze.mp4")
    fps = 25

    # first_batch_input, first_batch_real = next(iter(dataloader))
    # first_input = _post_compute(first_batch_input)[0]
    # first_real = _post_compute(first_batch_real)[0]
    # # with torch.no_grad(): first_fake = _post_compute(generator(first_batch_input.to(device)))[0]
    # first_fake = first_real
    width, height = get_wh(wl_video)

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
                    # wl_frame = cv2.cvtColor(wl_images[i], cv2.COLOR_RGB2BGR)
                    # color_frame = cv2.cvtColor(color_images[i], cv2.COLOR_RGB2BGR)
                    fake_frame = cv2.cvtColor(fake_images[i], cv2.COLOR_RGB2BGR)
                    fake_frame = cv2.resize(fake_frame, (width, height))
                    video_writer.write(fake_frame)
                print绿(f"\rProcessing batch: {k}", end='')
    finally:
        video_writer.release()
        print绿(f"Video saved at {output_video}")
    
    return os.path.abspath(output_video)
