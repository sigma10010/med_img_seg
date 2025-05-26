from .monuseg import MoNuSeg
from .liver_paip2019 import LiverC
from .fives import FIVES
from .tnbc import TNBC
from torch.utils import data

def get_loader(image_path, anno_path, batch_size, num_workers=3, size=256, data_aug=False, prob=0.5, num_slide=1, is_train = True, transform=None):
    """Builds and returns Dataloader."""
    # dataset = MoNuSeg(img_root = image_path, anno_root = anno_path, size=size, data_aug=data_aug, prob=prob, transform=transform)
    # dataset = TNBC(img_root = image_path, anno_root = anno_path, data_aug=data_aug, prob=prob, num_slide=num_slide, is_train=is_train, transform=transform)
    # dataset = LiverC(img_root = image_path, anno_root = anno_path, data_aug=data_aug, prob=prob, transform=transform)
    dataset = FIVES(img_root = image_path, anno_root = anno_path, data_aug=data_aug, prob=prob, transform=transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    return data_loader
