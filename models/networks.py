import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchvision import models
from torchvision import ops
import numpy as np

###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_Deep_Cell_Relationship(rel_classes, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = models.vgg16_bn(pretrained=True)
    net = Deep_Cell_Relationship(net, rel_classes)

    ## initalize the Deep_Cell_Relationship
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
    layer = 0
    for submodule in net.children():
        if layer == 0: continue
        layer += 1
        submodule.apply(init_func)
    
    print('initialize network')

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    
    return net

##############################################################################
# Classes
##############################################################################

class Deep_Cell_Relationship(nn.Module):
    def __init__(self, torchvision_model, rel_classes):
        super(Deep_Cell_Relationship, self).__init__()
        self.vgg16_layer = nn.Sequential(*list(torchvision_model.children())[:-2])
        #self.resnet_layer = nn.Sequential(*list(torchvision_model.children())[:-2])

        self.fc = nn.Sequential(*[nn.Linear(512*7*7, 4096), nn.ReLU(inplace=True), nn.Dropout(),
                                nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
                                nn.Linear(4096, 512), nn.ReLU(inplace=True)])

        self.fc_visual = nn.Sequential(*[nn.Linear(512*2, 512), nn.ReLU(inplace=True)])

        self.fc_spatial = nn.Sequential(*[nn.Linear(4*2, 256), nn.ReLU(inplace=True),
                                    nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(),
                                    nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(),])

        self.fc_cls = nn.Sequential(*[nn.Linear(1024, 512), nn.ReLU(inplace=True),
                                    nn.Linear(512, rel_classes), nn.ReLU(inplace=True)])            

    def forward(self, input, boxes, spatialFea, edges):
        # boxes: List[Tensor[L, 4]])
        cnn_feat = self.vgg16_layer(input)#[batch_size, 512, 16, 16]
        x_so = ops.roi_align(cnn_feat, boxes, 7) # [num_node, 512, 7, 7]
        x_so = self.fc(x_so.view(x_so.size(0), -1)) # [num_node, 512]

        x_s = torch.index_select(x_so, 0, edges[:,0])
        x_o = torch.index_select(x_so, 0, edges[:,1])
        visual_feat = torch.cat((x_s, x_o), 1)
        visual_feat = self.fc_visual(visual_feat) # 

        spatial_s = torch.index_select(spatialFea, 0, edges[:,0])
        spatial_o = torch.index_select(spatialFea, 0, edges[:,1])
        spatial_feat = torch.cat((spatial_s, spatial_o), 1)
        spatial_feat = self.fc_spatial(spatial_feat)

        fusion = torch.cat((visual_feat, spatial_feat), 1)
        real_score = self.fc_cls(fusion)

        return real_score
