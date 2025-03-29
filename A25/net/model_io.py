import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from A25.global_config import GlobalConfig as cfg
from A25.UTIL.colorful import *

def save_sup_model(gen, opt_G, pt_path,info=None):
    print绿('saving model to %s' % pt_path)
    torch.save({
        'gen_model': gen.state_dict(),
        'opt_G': opt_G.state_dict(),
    }, pt_path)

def load_sup_model_(gen,  opt_G, pt_path):
    if not os.path.exists(pt_path): 
        assert False, "file does not exists"

    cpt = torch.load(pt_path, map_location=cfg.device, weights_only=True)
    gen.load_state_dict(cpt['gen_model'], strict=True)
    opt_G.load_state_dict(cpt['opt_G'])
    print绿(f'loaded model {pt_path}')
    return gen,opt_G


def save_gan_model(gen, dis, opt_G, opt_D, pt_path,info=None):
    print绿('saving model to %s' % pt_path)
    torch.save({
        'gen_model': gen.state_dict(),
        'dis_model': dis.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_D': opt_D.state_dict(),
    }, pt_path)

def load_gan_model_(gen, dis, opt_G, opt_D, pt_path):
    if not os.path.exists(pt_path): 
        assert False, "file does not exists"

    cpt = torch.load(pt_path, map_location=cfg.device, weights_only=True)
    gen.load_state_dict(cpt['gen_model'], strict=True)
    dis.load_state_dict(cpt['dis_model'], strict=True)
    opt_G.load_state_dict(cpt['opt_G'])
    opt_D.load_state_dict(cpt['opt_D'])
    print绿(f'loaded model {pt_path}')
    return gen, dis, opt_G, opt_D


def load_gan_model(gen, pt_path):
    if not os.path.exists(pt_path): 
        assert False, "file does not exists"

    cpt = torch.load(pt_path, map_location=cfg.device, weights_only=True)
    gen.load_state_dict(cpt['gen_model'], strict=True)
    print绿(f'loaded model {pt_path}')
    return gen