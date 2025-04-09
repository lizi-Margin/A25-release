import torch, cv2, random
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from A25.global_config import GlobalConfig as cfg
from A25.UTIL.colorful import *
from A25.pre.extract_number import extract_number


train = False
def transform_pair(img1, img2):
    if not train:
        return img1, img2

    width, height = img1.shape[-2:]
    angle = random.uniform(-10, 10) 
    scale = random.uniform(0.95, 1.6) 
    translate_coef = 0.25
    translate = (int(random.uniform(-translate_coef, translate_coef) * width), int(random.uniform(-translate_coef, translate_coef) * height))
    hflip = random.random() > 0.5 

    img1 = F.affine(img1, angle=angle, translate=translate, scale=scale, shear=0)
    img2 = F.affine(img2, angle=angle, translate=translate, scale=scale, shear=0)

    if hflip:
        img1 = F.hflip(img1)
        img2 = F.hflip(img2)

    return img1, img2


class GanDataset(Dataset):
    def __init__(self, wl_dir, ir_dir, transform=None):
        self.wl_dir = wl_dir
        self.ir_dir = ir_dir
        self.transform = transform
        self.wl_files = sorted(os.listdir(wl_dir),key=extract_number)
        self.ir_files = sorted(os.listdir(ir_dir),key=extract_number)

    def __len__(self):
        return self.length()
    def length(self):
        # return int(min(len(self.wl_files), len(self.ir_files)) / 10)
        return min(len(self.wl_files), len(self.ir_files))

    def __getitem__(self, idx):
        return self.get(idx)
    
    def get(self, idx):
        wl_img_path = os.path.join(self.wl_dir, self.wl_files[idx])
        ir_img_path = os.path.join(self.ir_dir, self.ir_files[idx])
        assert os.path.isfile(wl_img_path)
        assert os.path.isfile(ir_img_path)
        wl_img = Image.open(wl_img_path).convert('RGB')
        ir_img = Image.open(ir_img_path).convert('RGB')

        if self.transform:
            wl_img = self.transform(wl_img)
            ir_img = self.transform(ir_img)
            wl_img, ir_img = transform_pair(wl_img, ir_img)
            assert isinstance(wl_img, torch.Tensor)
            assert isinstance(ir_img, torch.Tensor)  # RGB

            assert not torch.any(torch.isnan(wl_img)), str(wl_img)
            assert not torch.any(torch.isnan(ir_img)), str(ir_img)
        else:
            wl_img = np.array(wl_img)
            ir_img = np.array(ir_img)
            wl_img = cv2.cvtColor(wl_img, cv2.COLOR_RGB2BGR)  
            ir_img = cv2.cvtColor(ir_img, cv2.COLOR_RGB2BGR)   #BGR
        return wl_img, ir_img
    
class MemGanDataset(GanDataset):
    def __init__(self, wl_dir, ir_dir, transform=None):
        super().__init__(wl_dir, ir_dir, transform)
        self.mem = [self.get(i) for i in tqdm(range(self.length()), desc="Loading dataset into memory")]

    def __getitem__(self, idx):
        return self.mem[idx]

class DEYOLO_Dataset(GanDataset):
    def __init__(self, dir, wl_prefix='wl', ir_prefix='ir', transform=None):
        assert wl_prefix != ir_prefix
        self.wl_dir = dir
        self.ir_dir = dir
        
        all_files = sorted(os.listdir(dir),key=extract_number)
        self.wl_files = []
        self.ir_files = []

        for file in all_files:
            if file.startswith(wl_prefix):
                self.wl_files.append(file)
            if file.startswith(ir_prefix):
                self.ir_files.append(file)
        assert len(self.wl_files) == len(self.ir_files) and len(self.ir_files) > 0, f"{len(self.wl_files)}, {len(self.ir_files)}"

        self.transform = transform
        
class MemDEYOLO_Dataset(DEYOLO_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mem = [self.get(i) for i in tqdm(range(self.length()), desc="Loading dataset into memory")]

    def __getitem__(self, idx):
        return self.mem[idx]

    def __getitem__(self, idx):
        return self.mem[idx]



class MemVidGanDataset(Dataset):
    def __init__(self, wl_vid, ir_vid=None, transform=None):
        assert os.path.isfile(wl_vid)

        # self.wl_vid = wl_vid
        # self.ir_vid = ir_vid
        self.transform = transform
        self.wl_mem = self._extract_frames(wl_vid)

        if (self.transform is not None) and (ir_vid is not None):
            raise NotImplementedError("transformation of wl and ir is separated now")
        self.ir_mem = self._extract_frames(ir_vid) if (ir_vid is not None) else None
 
    
    def _extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc=f"Extracting frames from {video_path}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self._get(frame))
                pbar.update(1)
                if pbar.n  > 500: break
        cap.release()
        return frames
    
    def _get(self, array_frame):
        wl_img = Image.fromarray(array_frame)
        
        if self.transform:
            wl_img = self.transform(wl_img)
            wl_img, _ = transform_pair(wl_img, torch.zeros_like(wl_img))
            assert isinstance(wl_img, torch.Tensor)

            assert not torch.any(torch.isnan(wl_img)), str(wl_img)
        else:
            wl_img = np.array(wl_img)
            wl_img = cv2.cvtColor(wl_img, cv2.COLOR_RGB2BGR)  
        return wl_img


    def __len__(self):
        return len(self.wl_mem)
    
    def __getitem__(self, idx):
        wl = self.wl_mem[idx]
        if self.ir_mem is None:
            if isinstance(wl, torch.Tensor):
                ir = torch.zeros_like(wl).to(wl.device)
            elif isinstance(wl, np.ndarray):
                ir = np.zeros_like(wl)
            else: assert False
        else:
            ir = self.ir_mem[idx]
        return wl, ir
