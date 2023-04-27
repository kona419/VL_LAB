import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import torchvision
import torchvision.transforms as transforms
import warnings
import torch.optim as optim
import pdb

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "5" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

__all__ = ['crossstitch']


class StitchUnit(nn.Module):
    def __init__(self):
        super(StitchUnit, self).__init__()
        self.total_task = 4

        tensor_data = [[0.2 for col in range(self.total_task)] for row in range(self.total_task)]

        for i in range(len(tensor_data)):
            for j in range(len(tensor_data)):
                if i == j:
                    tensor_data[i][j] = 0.8

        self.stitch_matrix = nn.Parameter(data = torch.Tensor(tensor_data), requires_grad=True)

    def forward(self, x0, x1, x2): 
        x_data = [x0, x1, x2]
            
        for j in range(3):
            x_data[0] = x_data[0] + (self.stitch_matrix[0, j] * x_data[j])
            x_data[1] = x_data[1] + (self.stitch_matrix[1, j] * x_data[j])
            x_data[2] = x_data[2] + (self.stitch_matrix[2, j] * x_data[j])  ##

        return x_data[0], x_data[1], x_data[2]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()

        self.linear = nn.Linear(128 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Cross_stitch(nn.Module):
    def __init__(self):
        super(Cross_stitch, self).__init__()
        
        self.model0 = ResNet(BasicBlock, [2, 2, 2, 2],100)
        self.model1 = ResNet(BasicBlock, [2, 2, 2, 2],10)
        self.model2 = ResNet(BasicBlock, [2, 2, 2, 2],10)        
        
        self.cross_model_list = [self.model0, self.model1, self.model2]

        self.stitch_matrix1 = StitchUnit()
        self.stitch_matrix2 = StitchUnit()
        self.stitch_matrix3 = StitchUnit()
        self.stitch_matrix4 = StitchUnit()
        self.stitch_matrix5 = StitchUnit()
        
    def forward(self, data1, data2, data3):
        torch.autograd.set_detect_anomaly(True)

        x0 = data1
        x1 = data2
        x2 = data3
                
        x0 = F.relu(self.model0.bn1(self.model0.conv1(x0)))
        x1 = F.relu(self.model1.bn1(self.model1.conv1(x1)))
        x2 = F.relu(self.model2.bn1(self.model2.conv1(x2)))
      
        x0 = self.model0.layer1(x0)
        x1 = self.model1.layer1(x1)
        x2 = self.model2.layer1(x2)

        x0, x1, x2 = self.stitch_matrix1(x0, x1, x2)
                
        x0 = self.model0.layer2(x0)
        x1 = self.model1.layer2(x1)
        x2 = self.model2.layer2(x2)
           
        x0, x1, x2 = self.stitch_matrix2(x0, x1, x2)

        x0 = self.model0.layer3(x0)
        x1 = self.model1.layer3(x1)
        x2 = self.model2.layer3(x2)
            
        x0, x1, x2 = self.stitch_matrix3(x0, x1, x2)
        
        x0 = self.model0.layer4(x0)
        x1 = self.model1.layer4(x1)
        x2 = self.model2.layer4(x2)
            
        x0, x1, x2= self.stitch_matrix4(x0, x1, x2)

        x0 = self.model0.avgpool(x0)
        x1 = self.model1.avgpool(x1)
        x2 = self.model2.avgpool(x2)
        
        x0, x1, x2 = self.stitch_matrix5(x0, x1, x2)

        x0 = x0.view(x0.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x0 = self.model0.linear(x0)
        x1 = self.model1.linear(x1)
        x2 = self.model2.linear(x2)
   
        return x0, x1, x2


if __name__ == "__main__":
    
    warnings.filterwarnings('ignore')
    
    mnist_train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

    mnist_test_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))])


    transform_train_CIFAR = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test_CIFAR = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])



    cifar100_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_CIFAR)
    cifar100_trainloader = torch.utils.data.DataLoader(cifar100_trainset, batch_size=128, shuffle=True, drop_last=True)

    cifar100_testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test_CIFAR)
    cifar100_testloader = torch.utils.data.DataLoader(cifar100_testset, batch_size=128, shuffle=False, drop_last=True)
    
    cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_CIFAR)
    cifar10_trainloader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=128, shuffle=True, drop_last=True)

    cifar10_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_CIFAR)
    cifar10_testloader = torch.utils.data.DataLoader(cifar10_testset, batch_size=128, shuffle=False, drop_last=True)


    mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_train_transform)
    mnist_trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=128, shuffle=True, drop_last=True)

    mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_test_transform)
    mnist_testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=128, shuffle=False, drop_last=True)

    
    model = Cross_stitch().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total1,total2,total3 = 0,0,0
        correct1,correct2,correct3 = 0,0,0
        for i, ((data1, labels1), (data2, labels2), (data3, labels3)) in enumerate(zip(cifar100_trainloader, mnist_trainloader, cifar10_trainloader)):
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)


            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            labels3 = labels3.to(device)
            
            optimizer.zero_grad()
            
            output1, output2, output3 = model(data1, data2, data3)
            
            loss1 = criterion(output1, labels1)
            loss2 = criterion(output2, labels2)
            loss3 = criterion(output3, labels3)
     
            loss = loss1 + loss2 + loss3
         
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted1 = torch.max(output1.data, 1)
            total1 += labels1.size(0)
            correct1 += (predicted1 == labels1).sum().item()
            print('\nEpoch: %d' % epoch)
            print('cifar 100 Loss: %.3f | Acc: %.3f%%'
                        % (train_loss/(i+1), 100.*correct1/total1))
            
            _, predicted2 = torch.max(output2.data, 1)
            total2 += labels2.size(0)
            correct2 += (predicted2 == labels2).sum().item()
            print('\nEpoch: %d' % epoch)
            print('mnist Loss: %.3f | Acc: %.3f%%'
                        % (train_loss/(i+1), 100.*correct2/total2))
            
            _, predicted3 = torch.max(output3.data, 1)
            total3 += labels3.size(0)
            correct3 += (predicted3 == labels3).sum().item()
            print('\nEpoch: %d' % epoch)
            print('cifar 10 Loss: %.3f | Acc: %.3f%%'
                        % (train_loss/(i+1), 100.*correct3/total3))
            
    model.eval()
    total1, correct1 = 0, 0
    total2, correct2 = 0, 0
    total3, correct3 = 0, 0

    with torch.no_grad():
        for i, ((data1, labels1), (data2, labels2), (data3, labels3)) in enumerate(zip(cifar100_testloader, mnist_testloader, cifar10_testloader)):
            data1 = data1.to(device)
            labels1 = labels1.to(device)
                       
            data2 = data2.to(device)
            labels2 = labels2.to(device)
            
            data3 = data3.to(device)
            labels3 = labels3.to(device)
            
            output1, output2, output3 = model(data1, data2, data3)
            
            _, predicted1 = torch.max(output1.data, 1)
            total1 += labels1.size(0)
            correct1 += (predicted1 == labels1).sum().item()
            
            _, predicted2 = torch.max(output2.data, 1)
            total2 += labels2.size(0)
            correct2 += (predicted2 == labels2).sum().item()
            
            _, predicted3 = torch.max(output3.data, 1)
            total3 += labels3.size(0)
            correct3 += (predicted3 == labels3).sum().item()
            
        acc1 = 100 * correct1 /total1
        acc2 = 100 * correct2 /total2
        acc3 = 100 * correct3 /total3
        print('[test] CIFAR-100 accuracy : {:.3f}'.format(acc1))
        print('[test] MNIST accuracy : {:.3f}'.format(acc2))
        print('[test] CIFAR-10 accuracy : {:.3f}'.format(acc3))

        
    
