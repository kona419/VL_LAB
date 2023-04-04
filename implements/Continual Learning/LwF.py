import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from numpy import dsplit
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data.dataset import random_split
import torchvision.datasets as dataset
import os
import copy
import warnings 

from tqdm import tqdm 
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_schedule
from torch.autograd import Variable

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "4" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

#make a backbone model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, channel = [64, 128, 256, 512]):
        super(ResNet, self).__init__()
        self.in_planes = channel[0]
        self.last_planes = channel[3] *block.expansion

        self.conv1 = nn.Conv2d(3, channel[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, channel[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, channel[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channel[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, channel[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(self.last_planes, num_classes, bias = True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def feature_extraction(self, x): 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


class ModifiedResNet18(nn.Module):
    def __init__(self, make_model=True):
        super(ModifiedResNet18, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if make_model:
            self.make_model()
        

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedResNet18, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            
    def make_model(self):
        """Creates the model."""
        """NOTE: Choose ResNet model to use"""
        resnet = ResNet18()
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = nn.Sequential() 
        for name, module in resnet.named_children():
            if name != 'linear':
                self.shared.add_module(name, module)

        self.classifier = None
        
    def set_dataset(self, dataset):  # dataset name
        """Change the active classifier."""
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def add_dataset(self, dataset, num_outputs): 
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(512, num_outputs, bias = True))

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
  

def distillation_loss(y, teacher_scores, T, scale):
    return F.kl_div(F.log_softmax(y / T, dim = 1), F.softmax(teacher_scores / T, dim = 1)) * scale

class Manager(object):
    """Handles training and pruning."""

    def __init__(self, model, train_loader, test_loader, previous_model = None, device = device):    
        self.cuda = device
        self.previous_model = previous_model
        self.model = model

        self.optimizer = optim.SGD(self.model.parameters(), lr = 1e-2, momentum=0.9, weight_decay=1e-5)
        self.lr_schdule = lr_schedule.MultiStepLR(self.optimizer, milestones=[15, 20], gamma = 0.1)
        self.train_data_loader = train_loader
        self.test_data_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()

    def do_batch(self, batch, label, epoch_idx):
        """Runs model for one batch."""
        batch = batch.to(device)
        label = label.to(device)

        # Work with current model.
        self.model.zero_grad()
        x = self.model.shared(batch)
        x = x.view(x.size(0), -1)
        pred_logits = [classifier(x) for classifier in self.model.classifiers]
        # Apply cross entropy for current task.
        output = pred_logits[-1]
        new_loss = self.criterion(output, label)

        dist_loss = 0
        if self.previous_model is not None: 
            # Get targets using original model.
            self.previous_model.eval()
            x = self.previous_model.shared(batch)
            x = x.view(x.size(0), -1)
            target_logits = [classifier(x).data.to(device) for classifier in self.previous_model.classifiers]
            scale = [item.size(-1) for item in target_logits]
            
            # Compute loss.
            dist_loss = 0
            # Apply distillation loss to all old tasks.
            for idx in range(len(target_logits)):
                dist_loss += distillation_loss(pred_logits[idx], target_logits[idx], 2, scale[idx])

        loss = lambda_distill * dist_loss + new_loss
        loss.backward()

        if self.previous_model is not None: 
            #self.model.train_nobn()
            if epoch_idx <= ft_shared_after:
                # Set shared layer gradients to 0 
                for module in self.model.shared.modules():
                    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                        module.weight.grad.data.fill_(0)
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)

            # Set old classifier gradients to 0 
            for idx in range(len(self.model.classifiers)-1):
                module = self.model.classifiers[idx]
                module.weight.grad.data.fill_(0)
                if module.bias is not None:
                    module.bias.grad.data.fill_(0)

        # Update params.
        self.optimizer.step()
        return new_loss, lambda_distill * dist_loss

    def do_epoch(self, epoch_idx):
        """Trains model for one epoch."""
        for step, (image, label) in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            new_loss, dis_loss = self.do_batch(image, label, epoch_idx)
        self.lr_schdule.step()
        print('Epoch: {} | cla Loss: {:.3f} | dist Loss {:.3f}'.format(epoch_idx, new_loss, dis_loss))

    def train(self, epochs, savename=''):
        """Performs training."""
        self.model = self.model.to(self.cuda)
        
        for epoch in range(epochs):
            epoch_idx = epoch + 1
            self.model.train()
            self.do_epoch(epoch_idx)
            self.save_model(epoch_idx, savename)

    def eval(self):
        """Performs evaluation."""
        self.model.eval()
        total, correct = 0, 0 
        with torch.no_grad():
            for step, (batch, label) in enumerate(self.test_data_loader):
                batch = batch.to(device)
                label = label.to(device)
                output = self.model(batch)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)
            acc = 100 * correct /total

            print('[test] classification accuracy : {:.3f}'.format(acc))

            self.model.train()
            return acc

    def save_model(self, epoch, savename):
        """Saves model to file."""
        # Prepare the ckpt.
        ckpt = {
            'epoch': epoch,
            'model': self.model,
        }
        # Save to file.
        torch.save(ckpt, savename + '.pt') 


if __name__ == "__main__":
    
    warnings.filterwarnings('ignore')
    
    mnist_train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

    mnist_test_transform = transforms.Compose([
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




    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_CIFAR)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test_CIFAR)
            
    CIFAR_100_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 
    CIFAR_100_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_CIFAR)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_CIFAR)
            
    CIFAR_10_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 
    CIFAR_10_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_train_transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_test_transform)

    MNIST_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 
    MNIST_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft_shared_after = 10
    lambda_distill = 5.0
    epochs = 50

    dataset_list = ['CIFAR100', 'MNIST', 'CIFAR10']
    dataset_classes = [100, 10, 10]
    
    train_dataset_list = [CIFAR_100_train, MNIST_train, CIFAR_10_train]
    test_dataset_list = [CIFAR_100_test, MNIST_test, CIFAR_10_test]
    
    model = ModifiedResNet18().to(device)
    acc_dict = {}

    for task_idx in range(len(dataset_list)):
        print('{}-th task | dataset name : {}'.format(task_idx, dataset_list[task_idx]))
        previous_model = None 
        acc = []
        if task_idx > 0: 
            print('load trained model {}-th task | dataset name : {} '.format(task_idx-1, dataset_list[task_idx-1]))
            ckpt= torch.load(dataset_list[task_idx-1] + '.pt')
            model = ckpt['model']
            previous_model = copy.deepcopy(model)

        train_dataset, test_dataset = train_dataset_list[task_idx], test_dataset_list[task_idx]

        model.add_dataset(dataset_list[task_idx], dataset_classes[task_idx])
        model.set_dataset(dataset_list[task_idx])
        
        manager = Manager(model, train_dataset, test_dataset, previous_model=previous_model)
        manager.train(epochs, savename = dataset_list[task_idx])    
        
        print('check accuracy of previous tasks')
        for prev_task in range(task_idx + 1):
            print('previous task index {} | task name {}'.format(prev_task, dataset_list[prev_task]))
            model.set_dataset(dataset_list[prev_task])
            manager = Manager(model, train_dataset, test_dataset_list[prev_task], previous_model)
            result = manager.eval()
            acc.append(result)
        acc_dict[task_idx] = acc
           
