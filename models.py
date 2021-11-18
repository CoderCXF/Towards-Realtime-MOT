import os
from collections import defaultdict,OrderedDict

import torch.nn as nn

from utils.parse_config import *
from utils.utils import *
import time
import math

try:
    from utils.syncbn import SyncBN
    batch_norm=nn.BatchNorm2d  #SyncBN
except ImportError:
    batch_norm=nn.BatchNorm2d
# 该函数在Darknet读取YOLO网络之后被调用
# 主要作用是根据由yolov3_1088...cfg返回的嵌套多个dict的列表创建网络结构
def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)  # 超参是第一个元素width和height等等信息
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()   # pytorch的网络列表，自动注册
    yolo_layer_count = 0
    # 遍历每一个网络块
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            # 如果当前遍历到的块（moudle_def）是convolutional的话
            # 就往moudles中添加一个卷积层nn.Conv2d
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                after_bn = batch_norm(filters)
                modules.add_module('batch_norm_%d' % i, after_bn)
                # BN is uniformly initialized by default in pytorch 1.0.1. 
                # In pytorch>1.2.0, BN weights are initialized with constant 1,
                # but we find with the uniform initialization the model converges faster.
                nn.init.uniform_(after_bn.weight) 
                nn.init.zeros_(after_bn.bias)
            # 如果定义了一个LR的话，就往moudles中添加一个leakyReLU层
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        # 如果是maxpool层的话，就往moudles中添加一个maxpool
        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        # 如果是upsample的话，就往moudles中添加一个upsample
        elif module_def['type'] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        # 如果是route的话，就往moudles中添加一个route
        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())
        # 跨层连接
        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        # 如果是yolo的话
        elif module_def['type'] == 'yolo':
            # anchor_idxs = [0, 1, 2, 3]
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            # anchors = [(8,24), (11,34), (16,48), (23,68),   (32,96), (45,135), (64,192), (90,271),   (128,384), (180,540), 256,640, 512,640]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            # anchors = [(8, 24), (11, 34), (16, 48), (23, 68)]
            anchors = [anchors[i] for i in anchor_idxs]
            # nC = 1
            nC = int(module_def['classes'])  # number of classes
            img_size = (int(hyperparams['width']),int(hyperparams['height']))
            # Define detection layer and add moudle to moudles
            yolo_layer = YOLOLayer(anchors, nC, int(hyperparams['nID']), 
                                   int(hyperparams['embedding_dim']), img_size, yolo_layer_count)
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_layer_count += 1

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    # 最后返回超参以及moudle_list这个网络列表
    return hyperparams, module_list

# route and shorcut 层
class EmptyLayer(nn.Module):   # route层也可以看做的网络层，可以看做是线性输出的一层
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x

# 上采样层
class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

# 修改YOLOv3,核心代码，YOLO层
# YOLOv3 = Darknet5390(主干特征提取网络) + YOLOLayer
class YOLOLayer(nn.Module):
    '''
    anchors = [(8, 24), (11, 34), (16, 48), (23, 68)]
    nC = 1,only person
    nID =
    nE = hyperparams['embedding_dim']
    img_size = [1088, 608]
    '''
    def __init__(self, anchors, nC, nID, nE, img_size, yolo_layer):
        super(YOLOLayer, self).__init__()
        self.layer = yolo_layer
        nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (1)
        self.nID = nID # number of identities
        self.img_size = 0
        self.emb_dim = nE 
        self.shift = [1, 3, 5]

        self.SmoothL1Loss  = nn.SmoothL1Loss()  # L1损失：用于bbox回归
        self.SoftmaxLoss = nn.CrossEntropyLoss(ignore_index=-1)  # 交叉熵损失:用于分类
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)  # embedding loss
        self.s_c = nn.Parameter(-4.15*torch.ones(1))  # -4.15
        self.s_r = nn.Parameter(-4.85*torch.ones(1))  # -4.85
        self.s_id = nn.Parameter(-2.3*torch.ones(1))  # -2.3
        
        self.emb_scale = math.sqrt(2) * math.log(self.nID-1) if self.nID>1 else 1

    # YOLOLayer
    def forward(self, p_cat,  img_size, targets=None, classifier=None, test_emb=False):
        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]
        nB, nGh, nGw = p.shape[0], p.shape[-2], p.shape[-1]

        if self.img_size != img_size:
            create_grids(self, img_size, nGh, nGw)

            if p.is_cuda:
                self.grid_xy = self.grid_xy.cuda()
                self.anchor_wh = self.anchor_wh.cuda()

        p = p.view(nB, self.nA, self.nC + 5, nGh, nGw).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # permute
        # 特征图划分为三部分 13x13x...
        # 包含embedding信息的p_emb
        p_emb = p_emb.permute(0,2,3,1).contiguous()
        # 包含检测框位置信息的p_box
        p_box = p[..., :4]
        # 包含前景背景分类置信度的p_conf
        p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)  # Conf

        # Training and compute loss
        # 抽取监督信息
        if targets is not None:
            if test_emb:
                tconf, tbox, tids = build_targets_max(targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw)
            else:
                tconf, tbox, tids = build_targets_thres(targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw)
            tconf, tbox, tids = tconf.cuda(), tbox.cuda(), tids.cuda()
            mask = tconf > 0

            # Compute losses
            # lbox:检测框回归损失
            # lconf:前景背景分类损失
            # lid:embedding损失
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            nP = torch.ones_like(mask).sum().float()
            if nM > 0:
                lbox = self.SmoothL1Loss(p_box[mask], tbox[mask])
            else:
                FT = torch.cuda.FloatTensor if p_conf.is_cuda else torch.FloatTensor
                lbox, lconf =  FT([0]), FT([0])

            lconf =  self.SoftmaxLoss(p_conf, tconf)
            lid = torch.Tensor(1).fill_(0).squeeze().cuda()
            emb_mask,_ = mask.max(1)
            
            # For convenience we use max(1) to decide the id, TODO: more reseanable strategy
            tids,_ = tids.max(1) 
            tids = tids[emb_mask]
            embedding = p_emb[emb_mask].contiguous()
            embedding = self.emb_scale * F.normalize(embedding)
            nI = emb_mask.sum().float()
            
            if  test_emb:
                if np.prod(embedding.shape)==0  or np.prod(tids.shape) == 0:
                    return torch.zeros(0, self.emb_dim+1).cuda()
                emb_and_gt = torch.cat([embedding, tids.float()], dim=1)
                return emb_and_gt
            
            if len(embedding) > 1:
                logits = classifier(embedding).contiguous()
                lid =  self.IDLoss(logits, tids.squeeze())

            # Sum loss components 总损失
            loss = torch.exp(-self.s_r)*lbox + torch.exp(-self.s_c)*lconf + torch.exp(-self.s_id)*lid + \
                   (self.s_r + self.s_c + self.s_id)
            loss *= 0.5

            return loss, loss.item(), lbox.item(), lconf.item(), lid.item(), nT

        else:
            p_conf = torch.softmax(p_conf, dim=1)[:,1,...].unsqueeze(-1)
            p_emb = F.normalize(p_emb.unsqueeze(1).repeat(1,self.nA,1,1,1).contiguous(), dim=-1)
            #p_emb_up = F.normalize(shift_tensor_vertically(p_emb, -self.shift[self.layer]), dim=-1)
            #p_emb_down = F.normalize(shift_tensor_vertically(p_emb, self.shift[self.layer]), dim=-1)
            p_cls = torch.zeros(nB,self.nA,nGh,nGw,1).cuda()               # Temp
            p = torch.cat([p_box, p_conf, p_cls, p_emb], dim=-1)
            #p = torch.cat([p_box, p_conf, p_cls, p_emb, p_emb_up, p_emb_down], dim=-1)
            p[..., :4] = decode_delta_map(p[..., :4], self.anchor_vec.to(p))
            p[..., :4] *= self.stride

            return p.view(nB, -1, p.shape[-1])



class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_dict, nID=0, test_emb=False):
        super(Darknet, self).__init__()
        if isinstance(cfg_dict, str):
            cfg_dict = parse_model_cfg(cfg_dict)  # parse_model_cfg函数在utils目录下，返回的是一个列表（保存整个YOLOv3的网络信息）
        # moudle_defs保存整个网络的，列表中保存的是字典块（每一个卷积层是一块）
        self.module_defs = cfg_dict
        self.module_defs[0]['nID'] = nID
        self.img_size = [int(self.module_defs[0]['width']), int(self.module_defs[0]['height'])]
        self.emb_dim = int(self.module_defs[0]['embedding_dim'])  # 512
        self.hyperparams, self.module_list = create_modules(self.module_defs)  # create_moudles函数在此文件上方，此时整个网络已经构架完成
        self.loss_names = ['loss', 'box', 'conf', 'id', 'nT']
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        self.test_emb = test_emb
        
        self.classifier = nn.Linear(self.emb_dim, nID) if nID>0 else None



    def forward(self, x, targets=None, targets_len=None):
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        is_training = (targets is not None) and (not self.test_emb)
        #img_size = x.shape[-1]
        layer_outputs = []
        output = []
        '''
        moudle_defs : 列表，保存的是yolov3_1088..cfg中每一个层
        moudle_list : 已经创建好的网络模型
        '''
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':  # 维度拼接
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]   # 如果是1个的话，直接输出（相当于直连）
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)  #拼接
            # 跨层连接(残差块)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                if is_training:  # get loss
                    targets = [targets[i][:int(l)] for i,l in enumerate(targets_len)]
                    x, *losses = module[0](x, self.img_size, targets, self.classifier)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                elif self.test_emb:
                    if targets is not None:
                        targets = [targets[i][:int(l)] for i,l in enumerate(targets_len)]
                    x = module[0](x, self.img_size, targets, self.classifier, self.test_emb)
                else:  # get detections
                    x = module[0](x, self.img_size)
                    x = module[0](x, self.img_size)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            self.losses['nT'] /= 3 
            output = [o.squeeze() for o in output]
            return sum(output), torch.Tensor(list(self.losses.values())).cuda()
        elif self.test_emb:
            return torch.cat(output, 0)
        return torch.cat(output, 1)

def shift_tensor_vertically(t, delta):
    # t should be a 5-D tensor (nB, nA, nH, nW, nC)
    res = torch.zeros_like(t)
    if delta >= 0:
        res[:,:, :-delta, :, :] = t[:,:, delta:, :, :]
    else:
        res[:,:, -delta:, :, :] = t[:,:, :delta, :, :]
    return res 

def create_grids(self, img_size, nGh, nGw):
    self.stride = img_size[0]/nGw   # 倍数
    assert self.stride == img_size[1] / nGh, \
            "{} v.s. {}/{}".format(self.stride, img_size[1], nGh)

    # build xy offsets
    grid_x = torch.arange(nGw).repeat((nGh, 1)).view((1, 1, nGh, nGw)).float()
    grid_y = torch.arange(nGh).repeat((nGw, 1)).transpose(0,1).view((1, 1, nGh, nGw)).float()
    #grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4)

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)

# 权重文件有两种—— “.pt” 和 “.weights"结尾的，以”.pt"结尾的文件需要用 torch.load()来读取，
# 以 ".weights"结尾的文件需要用 load_darknet_weights()来读取
def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    # 如果没有预训练模型的话，需要从网上下载
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # 打开权重文件
    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    # 全卷积网络，只有卷积层和bn层有参数，由于bn层的参数是按照 bias, weight, running_mean, running_var的顺序写入列表的，
    # 所以读取的时候也应该按照这个顺序，同时由于有bn层的时候卷积层没有偏置，所以不用读取卷积层的偏置
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
