import torch, kornia, numpy as np, time
from torchvision.transforms import functional as F
from A25.third_party.AutomaticImageColorization.model import ColorNet
from A25.third_party.NighttimeImageEnhancement.illumination_boost_torch import illumination_boost
from A25.third_party.CEEF.CEEF import CEEF


color_model = ColorNet().cuda().eval()
color_model.load_state_dict(torch.load("./A25/third_party/AutomaticImageColorization/pretrained/checkpoint-trained.pth.tar", weights_only=True)['state_dict'])

import cv2
import A25.third_party.DehazeEnhanceQT.retinex as retinex
import A25.third_party.DehazeEnhanceQT.retinex_pytorch as retinex_pytorch
from json import load
with open('./A25/third_party/DehazeEnhanceQT/config.json', 'r') as f:
    config = load(f)

def batch_process(color_images: torch.Tensor, func):
    color_images = color_images.cpu().numpy().transpose((0, 2, 3, 1,))
    color_images = torch.from_numpy(np.array(
        [func(im) for im in color_images]
    ).transpose((0, 3, 1, 2,))).float().cuda()
    return color_images

def MSRCR(img):  # rgb input
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_msrcr = retinex.MSRCR(
        img,
        config['sigma_list'],
        config['G'],
        config['b'],
        config['alpha'],
        config['beta'],
        config['low_clip'],
        config['high_clip']
    )
    img_msrcr = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB) 
    return img_msrcr

def automatedMSRCR(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_msrcr = retinex.automatedMSRCR(
                    img,
                    config['sigma_list']
    )
    img_msrcr = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB) 
    return img_msrcr

def MSRCP(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_msrcr = retinex_pytorch.MSRCP(
        img,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']  
    )
    img_msrcr = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB) 
    return img_msrcr


def wl_to_color(wl_images):
    wl_images224 = F.resize(wl_images[:, 0:1, ...], (640, 640,))
    # wl_images224 = F.resize(wl_images[:, 0:1, ...], (224, 224,))
    wl_images224 = (wl_images224 + 1)/2
    with torch.no_grad(): color_images = color_model(wl_images224)
    
    # print(torch.std_mean(wl_images224), torch.min(wl_images224), torch.max(wl_images224))
    # print(torch.std_mean(color_images), torch.min(color_images), torch.max(color_images))
    color_images = torch.cat((wl_images224, color_images), 1)
    color_images[:, 0:1, ...] = color_images[:, 0:1, ...] * 100
    color_images[:, 1:3, ...] = color_images[:, 1:3, ...] * 255 - 128   
    color_images = kornia.color.lab_to_rgb(color_images)  # /255 is included in this func

    start = time.time()
    # color_images = illumination_boost(color_images, lambda_val=1.)
    # color_images = batch_process(color_images, func=CEEF); color_images = (color_images - color_images.min())/(color_images.max() - color_images.min())
    # color_images = batch_process(color_images, func=automatedMSRCR); color_images = (color_images - color_images.min())/(color_images.max() - color_images.min())
    # color_images = batch_process(color_images, func=MSRCR); color_images = (color_images - color_images.min())/(color_images.max() - color_images.min())
    color_images = batch_process(color_images, func=MSRCP); color_images = (color_images - color_images.min())/(color_images.max() - color_images.min())
    print(f"time={time.time() - start} s, size={color_images.shape[0]}")

    color_images = kornia.enhance.normalize(color_images, 0.5, 0.5)
    color_images = F.resize(color_images, wl_images.shape[-2:])
    return color_images



# from third_party.NIR2RGB.models import create_model
# from third_party.NIR2RGB.options.test_options import TestOptions
# opt = TestOptions().parse()  # get test options
# # hard-code some parameters for test
# opt.num_threads = 0   # test code only supports num_threads = 0
# opt.batch_size = 1    # test code only supports batch_size = 1
# opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
# opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
# opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
# nir2rgb_model = create_model(opt)      # create a model given opt.model and other options
# nir2rgb_model.setup(opt)               # regular setup: load and print networks; create schedulers
            # nir2rgb_data = {
            #     "A": wl_images,
            #     "B": wl_images,  # only to take the place
            #     "A_paths": None,
            #     "B_paths": None
            # }
            # nir2rgb_model.set_input(nir2rgb_data)  
            # nir2rgb_model.test()     
            # visuals = nir2rgb_model.get_current_visuals()  # get image results
            # img_path = nir2rgb_model.get_image_paths()     # get image paths
            # # print(img_path)