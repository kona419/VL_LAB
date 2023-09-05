import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

import numpy as np

import warnings
import os
import random
from collections import OrderedDict
from torch.optim.lr_scheduler import _LRScheduler

from torch.utils import data

from PIL import Image
from torchvision import transforms

from voc import VOCSegmentation
from resnet import resnet101
from ext_transforms import *

from metrics import StreamSegMetrics
from visualizer import Visualizer

import matplotlib
import matplotlib.pyplot as plt

model_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

#0902 하이퍼파라미터 : 배치 16&1 , crop_val :X, lr :0.01, output_stride:16, cropsize:512, multi-grid:(1,2,1)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "4,5,6,7" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = torch.nn.functional.upsample_bilinear(x, size=input_shape)
        return x
    
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            if self.hrnet_flag and name.startswith('transition'): # if using hrnet, you need to take care of transition
                if name == 'transition1': # in transition1, you need to split the module to two streams first
                    x = [trans(x) for trans in module]
                else: # all other transition is just an extra one stream split
                    x.append(module(x[-1]))
            else: # other models (ex:resnet,mobilenet) are convolutions in series.
                x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if name == 'stage4' and self.hrnet_flag: # In HRNetV2, we upsample and concat all outputs streams together
                    output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
                    x1 = torch.nn.functional.upsample_bilinear(x[1], size=(output_h, output_w))
                    x2 = torch.nn.functional.upsample_bilinear(x[2], size=(output_h, output_w))
                    x3 = torch.nn.functional.upsample_bilinear(x[3], size=(output_h, output_w))
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x
        return out
    
class DeepLabV3(_SimpleSegmentationModel):
    pass

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return torch.nn.functional.upsample_bilinear(x, size=size)
                
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    return model
                
def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet101(
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out'}
    classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def deeplabv3_resnet101(num_classes=21, output_stride=16, pretrained_backbone=True):
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]

def validate( model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    #if opts.save_val_results:
    if not os.path.exists('results'):
        os.mkdir('results')
        
    img_id = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            # if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
            #     ret_samples.append(
            #         (images[0].detach().cpu().numpy(), targets[0], preds[0]))
            
            denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            for i in range(len(images)):
                image = images[i].detach().cpu().numpy()
                target = targets[i]
                pred = preds[i]

                image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                target = loader.dataset.decode_target(target).astype(np.uint8)
                pred = loader.dataset.decode_target(pred).astype(np.uint8)

                Image.fromarray(image).save('results/%d_image.png' % img_id)
                Image.fromarray(target).save('results/%d_target.png' % img_id)
                Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                fig = plt.figure()
                plt.imshow(image)
                plt.axis('off')
                plt.imshow(pred, alpha=0.7)
                ax = plt.gca()
                ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                plt.close()
                img_id += 1

        score = metrics.get_results()
    return score

def main():

    warnings.filterwarnings('ignore')
    
    # Setup random seed
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    cur_itrs = 0
    cur_epochs = 0
    total_itrs = 30e3
    best_score = 0.0
    
    #data
    train_transform = ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            ExtRandomScale((0.5, 2.0)),
            ExtRandomCrop(size=(513, 513), pad_if_needed=True),
            ExtRandomHorizontalFlip(),
            ExtToTensor(),
            ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    
    test_transform = ExtCompose([
                ExtToTensor(),
                ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    
    train_dst = VOCSegmentation(root='./data', image_set='train', year='2012', download=True, transform=train_transform)
    val_dst = VOCSegmentation(root='./data', image_set='val', year='2012', download=True, transform=test_transform)

    train_loader = data.DataLoader(train_dst, batch_size=16, shuffle=True, num_workers=2, drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(val_dst, batch_size=1, shuffle=True, num_workers=2)
    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))
    sample = train_dst[0]
    print(type(sample[0]))
    print(type(sample[1]))
    
    #class 21개
    classes = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor")
    
    #model
    model = deeplabv3_resnet101(num_classes=21, output_stride=16).cuda()
   
    model = nn.DataParallel(model).to(device)
    set_bn_momentum(model.module.backbone, momentum=0.01)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    optimizer = torch.optim.SGD(params=[
        {'params': model.module.backbone.parameters(), 'lr': 0.1 * 0.01},
        {'params': model.module.classifier.parameters(), 'lr': 0.01},
    ], lr=0.01, momentum=0.9, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    scheduler = PolyLR(optimizer, total_itrs, power=0.9)
    
    metrics = StreamSegMetrics(21)
    
    # vis = Visualizer()
    # vis_sample_id = np.random.randint(0, len(val_loader), 8, np.int32)
    
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, path)
        print("Model saved as %s" % path)

    
    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            # if vis is not None:
            #     vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 20 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % 100 == 0:
                save_ckpt('/home/nayoung/nayoung/implement/DeepLab/checkpoints/latest444.pth')
                print("validation...")
                model.eval()
                val_score = validate(model=model, loader=val_loader, device=device, metrics=metrics)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    
                    best_score = val_score['Mean IoU']
                    file = open('/home/nayoung/nayoung/implement/DeepLab/best_result/best_text.txt', 'a')
                    file.write(str(best_score)+'\n')
                    file.close()
                    
                    save_ckpt('/home/nayoung/nayoung/implement/DeepLab/checkpoints/best444.pth')

                # if vis is not None:  # visualize validation score and samples
                #     vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                #     vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                #     vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                #     for k, (img, target, lbl) in enumerate(ret_samples):
                #         img = (denorm(img) * 255).astype(np.uint8)
                #         target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                #         lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                #         concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                #         vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()
            
            if cur_itrs >= total_itrs:
                return

if __name__ == '__main__':
    main()   