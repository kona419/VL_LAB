
from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

# Make custom dataset base on the CIFAR100 pytorch dataset
# custom dataset [class] must include three functions 
# __init__(self), __getitem__(index), and __len(self)__

__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam',
           'resnet152_cbam']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print('输入的shape为:'+str(x.shape))
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #print('avg_out的shape为:' + str(avg_out.shape))
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #print('max_out的shape为:' + str(max_out.shape))
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #out = self.ca(out) * out
        #out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

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

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        #y=torch.norm(x,p=2,dim=1,keepdim=True)
        #x/=y
        #x = self.fc(x)

        return x


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model

class iCIFAR100(CIFAR100):
    def __init__(self,root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iCIFAR100,self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas,labels=self.concatenate(datas,labels)
        self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
        self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
        print("the size of test set is %s"%(str(self.TestData.shape)))
        print("the size of test label is %s"%str(self.TestLabels.shape))


    def getTrainData(self,classes,exemplar_set):
        datas,labels=[],[]
        if len(exemplar_set)!=0:
            datas=[exemplar for exemplar in exemplar_set ]
            length=len(datas[0])
            labels=[np.full((length),label) for label in range(len(exemplar_set))]

        for label in range(classes[0],classes[1]):
            data=self.data[np.array(self.targets)==label]
            datas.append(data)
            labels.append(np.full((data.shape[0]),label))
        self.TrainData,self.TrainLabels=self.concatenate(datas,labels)
        print("the size of train set is %s"%(str(self.TrainData.shape)))
        print("the size of train label is %s"%str(self.TrainLabels.shape))

    def getTrainItem(self,index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

        if self.target_transform:
            target=self.target_transform(target)

        return index,img,target

    def getTestItem(self,index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img=self.test_transform(img)

        if self.target_test_transform:
            target=self.target_test_transform(target)

        return index, img, target

    def __getitem__(self, index):
        if self.TrainData!=[]:
            return self.getTrainItem(index)
        elif self.TestData!=[]:
            return self.getTestItem(index)


    def __len__(self):
        if self.TrainData!=[]:
            return len(self.TrainData)
        elif self.TestData!=[]:
            return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]
    
class network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)

    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self,inputs):
        return self.feature(inputs)

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class iCaRLmodel:

    def __init__(self,numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate):

        super(iCaRLmodel, self).__init__()
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.model = network(numclass,feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass
        self.transform = transforms.Compose([#transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None

        self.train_transform = transforms.Compose([#transforms.Resize(img_size),
                                                  transforms.RandomCrop((32,32),padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        self.test_transform = transforms.Compose([#transforms.Resize(img_size),
                                                   transforms.ToTensor(),
                                                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        self.classify_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                    #transforms.Resize(img_size),
                                                    transforms.ToTensor(),
                                                   transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        self.train_dataset = iCIFAR100('data', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('data', test_transform=self.test_transform, train=False, download=True)

        self.batchsize = batch_size
        self.memory_size=memory_size
        self.task_size=task_size

        self.train_loader=None
        self.test_loader=None

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        # numclass = the number of seen classes 
        # tasksize = the number of classes in the task. 
        classes=[self.numclass-self.task_size,self.numclass]
        self.train_loader,self.test_loader=self._get_train_and_test_dataloader(classes)
        if self.numclass>self.task_size:
            # expand the size of last fully connected layer of model
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        # if there is exemplar set, make joint training set with new data  
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader

    def train(self):
        # optimzier: mannually designed learning rate scheduler
         # First task (numclass == tasksize) - lr decay (lr x 0.2) at epoch : 48, 80
         # Other tasks (numclass> tasksize)  - lr decay (1r x 0.2) at epoch : 62 
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        for epoch in range(self.epochs):
            if epoch == 48:
                if self.numclass==self.task_size:
                     print(1)
                     opt = optim.SGD(self.model.parameters(), lr=1.0/5, weight_decay=0.00001)
                else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 5
                print("change learning rate:%.3f" % (self.learning_rate / 5))

            elif epoch == 62:
                if self.numclass>self.task_size:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 25
                else:
                     opt = optim.SGD(self.model.parameters(), lr=1.0/25, weight_decay=0.00001)
                print("change learning rate:%.3f" % (self.learning_rate / 25))

            elif epoch == 80:
                  if self.numclass==self.task_size:
                     opt = optim.SGD(self.model.parameters(), lr=1.0 / 125,weight_decay=0.00001)
                  else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 125

                  print("change learning rate:%.3f" % (self.learning_rate / 100))

            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
                # print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            accuracy = self._test(self.test_loader, 1)
            print('Epoch: {} | accuracy: {:.3f}'.format(epoch, accuracy))
        return accuracy

    def _test(self, testloader, mode):
        # There are two mode the test the iCaRL model.
        # 1) predcition based on : Nearest mean of exemplars [mode == 0]
        # 2) prediction based on : Fully connected layer [not implemented]
        if mode==0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy

    def _compute_loss(self, indexs, imgs, target):
        # binary cross entropy 
        # if first task : BCE with predicted output and target 
        # else: 
          # new data : BCE with predicted output and target 
          # old data : BCE with predicted output from current and previous model. (can use distillation loss)
        output=self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            old_target=torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)
        
    # change the size of examplar
    def afterTrain(self,accuracy):
        # if constructed exemplar size is larger than predetermined exemplar size --> reduce 
        self.model.eval()
        m=int(self.memory_size/self.numclass)
        self._reduce_exemplar_sets(m)
        for i in range(self.numclass-self.task_size,self.numclass):
            print('construct class %s examplar:'%(i),end='')
            images=self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images,m)
  
        self.numclass+=self.task_size
        self.compute_exemplar_class_mean()
        self.model.train()
        KNN_accuracy=self._test(self.test_loader,0)
        print("NMS accuracy : ", str(KNN_accuracy.item()))
        filename='accuracy:%.3f_KNN_accuracy:%.3f_increment:%d_net.pkl' % (accuracy, KNN_accuracy, i + 10)
        torch.save(self.model,filename)
        self.old_model=torch.load(filename)
        self.old_model.to(device)
        self.old_model.eval()
        
    
        
    def _construct_exemplar_set(self, images, m):
        # 1) calculate the mean value of features (class)
        # 2) select m samples per class 
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))
     
        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)
        #self.exemplar_set.append(images)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))


    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(device)
        feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        #feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print("compute the class mean of %s"%(str(index)))
            exemplar=self.exemplar_set[index]
            #exemplar=self.train_dataset.get_image_class(index)
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_,_=self.compute_class_mean(exemplar,self.classify_transform)
            class_mean=(class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)

    def classify(self, test):
        result = []
        test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        #test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)
    
if __name__ == "__main__":
    numclass=10
    feature_extractor=resnet18_cbam()
    img_size=32
    batch_size=128
    task_size=10
    memory_size=2000
    epochs=100
    learning_rate=2.0

    model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)
    #model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))

    acc = []
    plt.figure()
    for i in range(10):
        model.beforeTrain()
        accuracy=model.train()
        acc.append(accuracy)
        model.afterTrain(accuracy)
    #     plt.xlabel('task')
    #     plt.ylabel('Accuracy')
    #     plt.plot( i,accuracy,'ro')
    # plt.show()
