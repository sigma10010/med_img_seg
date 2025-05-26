import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from metrics.evaluation import *
from models.u_net import U_Net,R2U_Net,AttU_Net,R2AttU_Net,XXU_Net
from models.unet import UNet
from models.swin_unet import SwinTransformerSys
from models.transunet import TransUNetWithAttention
from models.vit_seg_modeling import VisionTransformer as ViT_seg
from models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from models.swinunet import SwinU

import csv
from losses.losses import MultiTaskLoss, FocalLoss, IoULoss, MultiLoss, GeneralizedL1Loss, NLLLoss
from losses.ssim import SSIMLoss, MS_SSIMLoss

METRICS = [MultiClassAccumulatedSPMetric(),\
            MultiClassAccumulatedPCMetric(),\
            MultiClassAccumulatedSEMetric(), \
            MultiClassAccumulatedJSMetric(), \
            MultiClassAccumulatedDCMetric()]

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, metrics = METRICS):
        # eveluation metric
        self.metrics = metrics

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
#         self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.size = config.input_size
        self.depth = config.depth
        self.width = config.width
        self.n_classes = config.n_classes
        self.category = config.category
        self.criterion = None
        self.augmentation_prob = config.augmentation_prob
        self.reduction_ratio = config.reduction_ratio
        self.n_skip = config.n_skip
        self.n_head = config.n_head
        self.is_shortcut = config.is_shortcut
        self.M = config.M
        self.is_scale_selective = config.is_scale_selective

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.start_epoch = config.start_epoch
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.fold = config.fold
        self.level = config.level
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.att_mode = config.att_mode
        self.conv_type = config.conv_type
        self.t = config.t
        self.build_model() # init self.unet
        
        # loss
        self.loss_type = config.loss_type
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.balance = config.balance
        self.init_loss() # init self.criterion
        self.ssim_loss = MS_SSIMLoss()

        self.unet_path = os.path.join(self.model_path, '%s-%s-level%s-size%s-depth%s-width%s-n_classes%s-nhead%s-fold%s.pkl' \
            %(self.model_type, self.loss_type, self.level, self.size, self.depth, \
            self.width, self.n_classes, self.n_head, self.fold))

        # torch.autograd.set_detect_anomaly(True)

    def build_model(self):
        """Build model""" 
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=3, n_classes = self.n_classes)
        elif self.model_type =='R2U_Net':
            self.unet = R2U_Net(img_ch=3,t=self.t, n_classes = self.n_classes)
        elif self.model_type =='AttU_Net':
            self.unet = AttU_Net(img_ch=3, n_classes = self.n_classes)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3,t=self.t, n_classes = self.n_classes)
        elif self.model_type in ['UNet','SAFS','SK','BAM','SE','CBAM', 'BAM-SAFS','CBAM-SAFS','SE-SAFS','SK-SAFS', 'SK-BAM-SAFS', 'SK-SE-SAFS']:
            self.unet = UNet(n_classes=self.n_classes, \
                init_features=self.width, \
                reduction_ratio=self.reduction_ratio, \
                n_head = self.n_head, \
                att_mode = self.att_mode, \
                is_scale_selective = self.is_scale_selective, \
                is_shortcut = self.is_shortcut, \
                conv_type = self.conv_type)
        elif self.model_type =='SwinUnet':
            # self.unet = SwinTransformerSys(img_size=self.size, num_classes = self.n_classes)
            self.unet = SwinU()
        elif self.model_type =='TransUnet':
            # self.unet = TransUNetWithAttention(in_ch=3, out_ch=self.n_classes)
            vit_name ='ViT-B_16'
            config_vit = CONFIGS_ViT_seg[vit_name]
            config_vit.n_classes = 2
            config_vit.n_skip = 0
            self.unet = ViT_seg(config_vit)
        else:
            raise NotImplementedError(self.model_type+" is not implemented")

        # self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, [self.beta1, self.beta2])
        self.optimizer = optim.Adam(
            list(self.unet.parameters()), 
            lr=self.lr,          
            betas=(self.beta1, self.beta2), 
            weight_decay=1e-4        # example L2 regularization strength
        )

        self.unet.to(self.device)

#         self.print_network(self.unet, self.model_type)

    def init_loss(self):
        """instantiate loss function."""
        if self.loss_type =='l1':
            self.criterion = GeneralizedL1Loss(alpha=self.alpha, gamma=self.gamma, beta = self.balance)
        elif self.loss_type =='nll':
            self.criterion = torch.nn.NLLLoss()
        elif self.loss_type =='ssim':
            self.criterion = MS_SSIMLoss()
        elif self.loss_type =='iou':
            self.criterion = IoULoss()
        elif self.loss_type =='focal':
            self.criterion = FocalLoss(gamma=self.gamma, alpha=self.balance)
        elif self.loss_type =='multitask':
            self.criterion = MultiTaskLoss(torch.nn.NLLLoss(), torch.nn.NLLLoss())
        elif self.loss_type =='nll+iou':
            self.criterion = MultiLoss([torch.nn.NLLLoss(), IoULoss()], names = ['NLLLoss', 'IoULoss'], weights = [1, 1])
        elif self.loss_type =='nll+ssim':
            self.criterion = MultiLoss([torch.nn.NLLLoss(), MS_SSIMLoss()], names = ['NLLLoss', 'SSIMLoss'], weights = [1, 1])
        elif self.loss_type =='nll+ssim+iou':
            self.criterion = MultiLoss([torch.nn.NLLLoss(), MS_SSIMLoss(), IoULoss()], names = ['NLLLoss', 'SSIMLoss', 'IoULoss'], weights = [1, 1, 1])
        else:
            raise NotImplementedError(self.loss_type+" is not implemented")

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()
        
    def train(self, save_loss = False):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#

        unet_path_last = self.unet_path[0:-4]+'-%s.pkl'%self.start_epoch
        print(self.unet_path)
        print(unet_path_last)

        # U-Net Train
        if os.path.isfile(unet_path_last):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path_last))
            print('%s is Successfully Loaded from %s'%(self.model_type,unet_path_last))

        # Train for Encoder
        lr = self.lr
        best_unet_score = 0.
        
        losses_disc = {} # save losses to plot
        
        for epoch in range(self.start_epoch, self.num_epochs):
            self.unet.train(True)
            epoch_losses = {}

            # reset metrics
            for metric in self.metrics:
                metric.reset()

            for i, (images, GT) in enumerate(self.train_loader):
                # print(f'{i+1}/{len(self.train_loader)}')
                # GT : Ground Truth tupple (mask, label)

                images = images.to(self.device)
                mask = GT[0]
                mask = mask.squeeze(1).type(torch.LongTensor).to(self.device)
                label = GT[1].type(mask.type())
                label = label.to(self.device)

                # SR : Segmentation Result
                # CR : Classification Result
                # output_dict = self.unet(images)
                # SR = output_dict['seg_output']
                # SR_dual = output_dict['seg_dual']
                SR, CR = self.unet(images)

                # print(SR.shape, mask.shape)
                # print(mask.min(), mask.max())
                loss_outputs = self.criterion(SR,mask)
                # loss_dual = self.criterion(SR_dual,mask)
                
                if type(loss_outputs) is dict:
                    loss = loss_outputs['total_loss']
                    # loss = loss_outputs['total_loss']+loss_dual['total_loss']
                    for key in loss_outputs.keys():
                        loss_key = loss_outputs[key]
                        if key not in epoch_losses.keys():
                            epoch_losses[key]=loss_key.item()
                        else:
                            epoch_losses[key]+=loss_key.item()
                else:
                    loss = loss_outputs
                    # loss = loss_outputs+loss_dual
                    key = 'total_loss'
                    if key not in epoch_losses.keys():
                        epoch_losses[key]=loss.item()
                    else:
                        epoch_losses[key]+=loss.item()

                # Backprop + optimize
                with torch.autograd.set_detect_anomaly(True):
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                for metric in self.metrics:
                    # cls metric: ACC F1
                    if metric.name()=='Accuracy' or metric.name()=='F1':
                        print('Accuracy')
                        # metric(CR,label)
                    # seg metric: JS DC
                    else:
                        metric(SR,mask, label = self.category)
                        # metric(SR,mask)

            # Print the log info
            message = 'Epoch: {}/{}. Train set: '.format(epoch+1, self.num_epochs)
            for key in epoch_losses.keys():
                message += ' {}: {:.4f}'.format(key, epoch_losses[key]/len(self.train_loader)*100)
                key_train = key+'_train'
                if key_train not in losses_disc.keys():
                    losses_disc[key_train]=[epoch_losses[key]/len(self.train_loader)*100]
                else:
                    losses_disc[key_train].append(epoch_losses[key]/len(self.train_loader)*100)
            for metric in self.metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            print(message)

            # Decay learning rate
            if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print ('Decay learning rate to lr: {}.'.format(lr))

            #===================================== Validation ====================================#
            self.unet.train(False)
            self.unet.eval()

            # reset metrics
            for metric in self.metrics:
                metric.reset()
                
            epoch_losses = {}
            for i, (images, GT) in enumerate(self.valid_loader):
                # print(f'{i+1}/{len(self.valid_loader)}')
                mask = GT[0]
                mask = mask.squeeze(1).type(torch.LongTensor).to(self.device)
                label = GT[1].type(mask.type())
                label = label.to(self.device)
                images = images.to(self.device)

                SR, CR = self.unet(images)
                # output_dict = self.unet(images)
                # SR = output_dict['seg_output']
                label = label.view(label.size(0),-1)
                
                loss_outputs = self.criterion(SR,mask)
                # consider multi-loss
                if type(loss_outputs) is dict:
                    loss = loss_outputs['total_loss']
                    for key in loss_outputs.keys():
                        loss_key = loss_outputs[key]
                        if key not in epoch_losses.keys():
                            epoch_losses[key]=loss_key.item()
                        else:
                            epoch_losses[key]+=loss_key.item()
                else:
                    loss = loss_outputs
                    key = 'total_loss'
                    if key not in epoch_losses.keys():
                        epoch_losses[key]=loss.item()
                    else:
                        epoch_losses[key]+=loss.item()

                for metric in self.metrics:
                    # cls metric: ACC F1
                    if metric.name()=='Accuracy' or metric.name()=='F1':
                        print('Accuracy')
                        # metric(CR,label)
                    # seg metric: JS DC
                    else:
                        metric(SR,mask, label = self.category)
                        # metric(SR,mask)

            # Print the log info
            message = 'Epoch: {}/{}. Valid set: '.format(epoch+1, self.num_epochs)
            for key in epoch_losses.keys():
                message += ' {}: {:.4f}'.format(key, epoch_losses[key]/len(self.valid_loader)*100)
                key_test = key+'_test'
                if key_test not in losses_disc.keys():
                    losses_disc[key_test]=[epoch_losses[key]/len(self.valid_loader)*100]
                else:
                    losses_disc[key_test].append(epoch_losses[key]/len(self.valid_loader)*100)
            unet_score = 0
            for i, metric in enumerate(self.metrics):
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                if metric.name()=='JS' or metric.name()=='DC': # only add seg metric
                    unet_score+=float(metric.value())
            print(message)


            # Save Best U-Net model
            if unet_score > best_unet_score:
                best_unet_score = unet_score
                best_unet = self.unet.state_dict()
                print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
                torch.save(best_unet,self.unet_path)
                
            # Save last epoch U-Net model
            if (epoch+1)==self.num_epochs:
                final_unet = self.unet.state_dict()
                unet_path_final = self.unet_path[0:-4]+'-%s.pkl'%(epoch+1)
                torch.save(final_unet,unet_path_final)
        print('Best %s model score : %.4f'%(self.model_type,best_unet_score))

        # save losses against epochs after training
        if save_loss:
            for key in losses_disc.keys():
                filename = self.unet_path[0:-4]+'-'+key+'.npy'
                np.save(filename, np.array(losses_disc[key]))
                

        #===================================== Test after finishing training====================================#
        del self.unet
        del best_unet
        print('Test after finishing training')
        self.test()
        
    def test(self):
        print(self.unet_path)
        self.build_model()
        self.unet.load_state_dict(torch.load(self.unet_path))

        self.unet.train(False)
        self.unet.eval()


        # reset metrics
        for metric in self.metrics:
            metric.reset()
        with torch.no_grad():
            for i, (images, GT) in enumerate(self.valid_loader):
                # print(f'{i+1}/{len(self.valid_loader)}')
                mask = GT[0]
                mask = mask.to(self.device)
                label = GT[1].type(mask.type())
                label = label.to(self.device)
                images = images.to(self.device)

                SR, CR = self.unet(images)
                # output_dict = self.unet(images)
                # SR = output_dict['seg_output']
                label = label.view(label.size(0),-1)

                for metric in self.metrics:
                    # cls metric: ACC F1
                    if metric.name()=='Accuracy' or metric.name()=='F1':
                        print('Accuracy')
                        # metric(CR,label)
                    # seg metric: JS DC
                    else:
                        metric(SR,mask, label = self.category)
                        # metric(SR,mask)

        with open(os.path.join(self.result_path, 'result_FIVES.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            print([float(metric.value()) for metric in self.metrics])
            wr.writerow([self.model_type, self.loss_type, self.level, self.size, self.depth, self.width, \
                self.n_classes, self.alpha, self.gamma, self.n_head, self.fold]+[float(metric.value()) for metric in self.metrics])