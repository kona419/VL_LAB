import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import numpy as np
import math
import warnings
import os
import random

import utils

from PIL import Image

import torchvision.transforms as transforms
from torchvision import transforms

from torchvision.datasets import VOCSegmentation

from metrics import StreamSegMetrics
from visualizer import Visualizer

import matplotlib
import matplotlib.pyplot as plt

model_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

class Atrous_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Atrous_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Atrous_ResNet_features(nn.Module):

    def __init__(self, block, layers, pretrained=False):
        super(Atrous_ResNet_features, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, rate=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, rate=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, rate=1)
        self.layer4 = self._make_MG_unit(block, 512, stride=1, rate=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            print('load the pre-trained model.')
            resnet = models.resnet101(pretrained)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x 

class Atrous_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)

        return x

class DeepLabv3(nn.Module):
    def __init__(self, num_classes, small=True, pretrained=False):
        super(DeepLabv3, self).__init__()
        block = Atrous_Bottleneck
        self.resnet_features = Atrous_ResNet_features(block, [3, 4, 23], pretrained)

        rates = [1, 6, 12, 18]
        self.aspp1 = Atrous_module(2048 , 256, rate=rates[0])
        self.aspp2 = Atrous_module(2048 , 256, rate=rates[1])
        self.aspp3 = Atrous_module(2048 , 256, rate=rates[2])
        self.aspp4 = Atrous_module(2048 , 256, rate=rates[3])
        self.image_pool = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Conv2d(2048, 256, kernel_size=1))

        self.fc1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1),
                                 nn.BatchNorm2d(256))
        self.fc2 = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.resnet_features(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.image_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='nearest')

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.upsample(x, scale_factor=(16,16), mode='bilinear')

        return x 
    
# class VOCSegDataset(VOCSegmentation):
#     def __getitem__(self, idx):
#         image = Image.open(self.images[idx]).convert('RGB')
#         label = Image.open(self.masks[idx])

#         if self.transforms is not None:
#             seed = random.randint(0, 2 ** 32)
#             self._set_seed(seed); image = self.transforms(image)
#             self._set_seed(seed); label = self.transforms(label) * 255
#             label[label > 20] = 0

#         return image, label

#     def _set_seed(self, seed):
#         random.seed(seed)
#         torch.manual_seed(seed)


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    
    #data
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(512, (0.5, 2.0)),
        #   transforms.RandomCrop(512, padding=4), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    
    test_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    
    target_transform = target_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
    
    trainset = VOCSegmentation(root='./data', image_set='train', year='2012', download=True, transform=train_transform, target_transform=target_transform)
    testset = VOCSegmentation(root='./data', image_set='val', year='2012', download=True, transform=test_transform, target_transform=target_transform)

    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0) 
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
    print("Train set: %d, Val set: %d" % (len(trainset_loader), len(testset_loader)))
    # sample = trainset[0]
    # print(type(sample[0]))
    # print(type(sample[1]))
    
    #class 21개
    classes = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor")
    
    #model
    model = DeepLabv3(21).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    
    metrics = StreamSegMetrics(21)
    
    cur_itrs = 0
    cur_epochs = 0
    output_stride = 16
    epochs = 1
    
    enable_vis = False
    
    vis = Visualizer(port=13570, env="main") if enable_vis else None
    
    # Setup random seed
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    def train(epoch):
        model.train()
        interval_loss = 0
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(trainset_loader):
            batch_idx += 1
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            labels = torch.squeeze(labels, 1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #np_loss = loss.detach().cpu().numpy()
            interval_loss += loss.item()
            
            interval_loss = interval_loss / 10
            if batch_idx%50 == 0:
                print("Epoch %d, Itrs %d, Loss=%f" % (epoch, batch_idx, interval_loss))
            #interval_loss = 0.0
            
            # train_loss+= loss.item()
            # _, predicted = outputs.max(1)
            # total += labels.size(0)
            # correct += predicted.eq(labels).sum().item()
            # print('\nEpoch: %d' % epoch)
            # print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    def test(epoch, best_score):
        save_ckpt('/home/nayoung/nayoung/implement/DeepLab/checkpoints/latest_%s_os%d.pth' % ("PascalVOC22", output_stride))
        print("validation...")
        model.eval()
        
        # val_score, ret_samples = validate(
        #     opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
        #     ret_samples_ids=vis_sample_id)
        
        metrics.reset()
        ret_samples = []
        
        if not os.path.exists('results'):
            os.mkdir('results')
        
        vis_sample_id = np.random.randint(0, len(testset), 8, np.int32) if enable_vis else None  # sample idxs for visualization
        img_id = 0
        test_loss = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(testset_loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                labels = torch.squeeze(labels, 1)

                outputs = model(images)
                
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, prediction = outputs.max(1)
                
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)
                
                #metrics.synch(device)
            val_score = metrics.get_results()
                
                # if vis_sample_id is not None and i in vis_sample_id:  # get vis samples
                #     ret_samples.append(
                #         (images[0].detach().cpu().numpy(), targets[0], preds[0]))

                #if opts.save_val_results:
                # for i in range(len(images)):
                #     image = images[i].detach().cpu().numpy()
                #     target = targets[i]
                #     pred = preds[i]

                #     image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                #     # target = testset_loader.dataset(target).astype(np.uint8)
                #     # pred = testset_loader.dataset(pred).astype(np.uint8)

                #     Image.fromarray(image).save('results/%d_image.png' % img_id)
                #     Image.fromarray(target).save('results/%d_target.png' % img_id)
                #     Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                #     fig = plt.figure()
                #     plt.imshow(image)
                #     plt.axis('off')
                #     plt.imshow(pred, alpha=0.7)
                #     ax = plt.gca()
                #     ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                #     ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                #     plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                #     plt.close()
                #     img_id += 1

            #val_score = metrics.get_results()
        
        
        print(metrics.to_str(val_score))
        # print(best_score)
        # if val_score['Mean IoU'] > best_score:  # save best model
        #     best_score = val_score['Mean IoU']
        #     save_ckpt('/home/nayoung/nayoung/implement/DeepLab/checkpoints/best_%s_os%d.pth' % ("PascalVOC", output_stride))

        # if vis is not None:  # visualize validation score and samples
        #     vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
        #     vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
        #     vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

            # for k, (img, target, lbl) in enumerate(ret_samples):
            #     img = (denorm(img) * 255).astype(np.uint8)
            #     target = trainset_loader(target).transpose(2, 0, 1).astype(np.uint8)
            #     lbl = testset_loader(lbl).transpose(2, 0, 1).astype(np.uint8)
            #     concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
            #     vis.vis_image('Sample %d' % k, concat_img)
                
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
        
    def denorm(image, mean=(0.485, 0.456, 0.4069), std=(0.229, 0.224, 0.225)):
        image = np.copy(image)
        if image.ndim == 3:
            assert image.ndim == 3, "Expected image [CxHxW]"
            assert image.shape[0] == 3, "Expected RGB image [3xHxW]"

            # for t, m, s in zip(image, mean, std):
            #     t.mul_(s).add_(m)
            for t in range(3):
                image[t, :, :] = image[t, :, :] * std[t] + mean[t]
            
        elif image.ndim == 4:
            # batch mode
            assert image.shape[1] == 3, "Expected RGB image [3xHxW]"

            # for t, m, s in zip((0, 1, 2), mean, std):
            #     image[:, t, :, :].mul_(s).add_(m)
            for t in range(3):
                image[:, t, :, :] = image[:, t, :, :] * std[t] + mean[t]

        return image

    best_score = 0
    for epoch in range(epochs):
        train(epoch)
        
        scheduler.step()
    test(epoch, best_score)