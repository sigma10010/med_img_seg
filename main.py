import argparse
import os
from solver import Solver
from data.data_loader import get_loader
# from data.data_split import divide_data
from torch.backends import cudnn
from torchvision import transforms

import torch
import numpy as np
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Disable optimizations that introduce randomness
# torch.use_deterministic_algorithms(True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(config):
    cudnn.benchmark = True
#     if config.model_type not in ['XXU_Net', 'U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
#         print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
#         print('Your input for model_type was %s'%config.model_type)
#         return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
#     config.result_path = os.path.join(config.result_path,config.model_type)
#     if not os.path.exists(config.result_path):
#         os.makedirs(config.result_path)
    
#     lr = random.random()*0.0005 + 0.0000005
#     augmentation_prob= random.random()*0.7
#     epoch = random.choice([100,150,200,250])
#     decay_ratio = random.random()*0.8
#     decay_epoch = int(epoch*decay_ratio)

#     config.augmentation_prob = augmentation_prob
#     config.num_epochs = epoch
#     config.lr = lr
#     config.num_epochs_decay = decay_epoch

    print(config)
    
    # divide data for k-fold cross validation
    # dd = divide_data(config.root_path)
    # dd.reset() # move data to all/
    # dd.divide(config.fold)
        
    train_loader = get_loader(image_path=config.train_path,
                            anno_path=config.train_anno_path,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            size=config.input_size,
                            data_aug=True,
                            prob=config.augmentation_prob,
                            num_slide=config.fold,
                            transform=transforms.Compose([
                                transforms.Resize(config.input_size),
                                transforms.ToTensor()
                             ]))
    valid_loader = get_loader(image_path=config.valid_path,
                            anno_path=config.valid_anno_path,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            size=config.input_size,
                            data_aug=False,
                            prob=0.,
                            num_slide=config.fold,
                            is_train = False,
                            transform=transforms.Compose([
                                transforms.Resize(config.input_size),
                                transforms.ToTensor()
                             ]))
    # for monuseg
    # test_loader = get_loader(image_path=config.test_path,
    #                         anno_path=config.test_path,
    #                         batch_size=config.batch_size,
    #                         num_workers=config.num_workers,
    #                         data_aug=False,
    #                         prob=0.,
    #                         num_slide=config.fold,
    #                         transform=transforms.Compose([
    #                             transforms.Resize(config.image_size),
    #                             transforms.ToTensor()
    #                          ]))

    solver = Solver(config, train_loader, valid_loader)
    # solver = Solver(config, train_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    # model
    parser.add_argument('--depth', type=int, default=5, help='#conv blocks, 512-256-128-64-32')
    parser.add_argument('--width', type=int, default=32, help='#channel, 32-64-128-256-512')
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--category', type=int, default=1, help='category for evaluation label')
    parser.add_argument('--t', type=int, default=2, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    parser.add_argument('--reduction_ratio', type=int, default=None, help='reduction ratio for attention layer') 
    parser.add_argument('--n_skip', type=int, default=4, help='number of skip-connection layers, <= depth-1') 
    parser.add_argument('--n_head', type=int, default=3, help='number of heads for prediction, 1 <= depth-1') 
    parser.add_argument('--att_mode', type=str, default='bam', help='cbam/bam/se') 
    parser.add_argument('--conv_type', type=str, default='basic', help='basic/sk')
    parser.add_argument('--is_shortcut', type=str2bool, default=True)
    parser.add_argument('--M', type=int, default=2, help='# kernel for SK net')
    parser.add_argument('--is_scale_selective', type=str2bool, default=True)
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.5)
    parser.add_argument('--start_epoch', type=int, default=0)
    
    # log
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--val_step', type=int, default=2)
    
    # loss
    parser.add_argument('--loss_type', type=str, default='nll', help='l1/nll/focal/iou/dice/multitask/nll+iou/nll+ssim/...')
    parser.add_argument('--alpha', type=float, default=1)        # alpha for l1 loss
    parser.add_argument('--gamma', type=float, default=1)        # gamma for l1/Focal loss
    parser.add_argument('--balance', type=list, default=None)   # balance factor

    # misc
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model_type', type=str, default='U_Net', help='XXU_Net/U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./ckpts/')
    parser.add_argument('--root_path', type=str, default='./datasets/TNBC_NucleiSegmentation/slide/')
    parser.add_argument('--train_path', type=str, default='./datasets/FIVES/512/train/Original/')
    parser.add_argument('--train_anno_path', type=str, default='./datasets/FIVES/512/train/Groundtruth/')
    parser.add_argument('--valid_path', type=str, default='./datasets/FIVES/512/test/Original/')
    parser.add_argument('--valid_anno_path', type=str, default='./datasets/FIVES/512/test/Groundtruth/')
    # parser.add_argument('--train_path', type=str, default='./datasets/monuseg/original_training/patches/s250/')
    # parser.add_argument('--train_anno_path', type=str, default='./datasets/monuseg/original_training/patches/s250/')
    # parser.add_argument('--valid_path', type=str, default='./datasets/monuseg/original_testing/patches/s250/')
    # parser.add_argument('--valid_anno_path', type=str, default='./datasets/monuseg/original_testing/patches/s250/')
    parser.add_argument('--test_path', type=str, default='./datasets/monuseg/original_testing/tissue_Images/')
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--fold', type=int, default=1, help='cross validation')
    parser.add_argument('--level', type=int, default=2, help='1/2')
    
    # other
    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    # config.root_path = '/mnt/DATA_OTHER/paip2019/patches/seg/level%d_%d/'%(config.level, config.image_size)
    # config.train_path = '/mnt/DATA_OTHER/paip2019/patches/seg/level%d_%d/train/'%(config.level, config.image_size)
    # config.valid_path = '/mnt/DATA_OTHER/paip2019/patches/seg/level%d_%d/validation/'%(config.level, config.image_size)
    if config.n_skip>config.depth-1:
        config.n_skip = config.depth-1
    
    main(config)
