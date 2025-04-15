import torchvision.transforms as transforms
import torch

class FillNaNWithZero:
    def __call__(self, tensor):
        tensor[torch.isnan(tensor)] = 0
        return tensor

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # FillNaNWithZero()
])



transform_dwgan = transforms.Compose([
    # transforms.Resize((320, 320)),
    transforms.Resize((256, 256)),
    # transforms.Resize((560, 752)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # FillNaNWithZero()
])
