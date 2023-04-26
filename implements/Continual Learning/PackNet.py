import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
import copy
import os

import warnings 

import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "5" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

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

class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, channel = [16,32,64]): ## 16, 32, 64
        super(ResNet_cifar, self).__init__()
        self.in_planes = channel[0]
        self.last_planes = channel[2] *block.expansion

        self.conv1 = nn.Conv2d(3, channel[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel[0])
        self.layer1 = self._make_layer(block, channel[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, channel[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channel[2], num_blocks[2], stride=2)
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
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def feature_extraction(self, x): 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
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

        self.shared = nn.Sequential() 
        for name, module in resnet.named_children():
            if name != 'linear' and name != 'avgpool':
                self.shared.add_module(name, module)

        self.batchnorms = nn.ModuleList()
        self.classifier = None
        self.batchnorm = None
        
    def set_dataset(self, dataset): 
        """Change the active classifier."""
        self.classifier = self.classifiers[self.datasets.index(dataset)]
        self.batchnorm = self.batchnorms[self.datasets.index(dataset)]

        for name, module in enumerate(self.shared.modules()): 
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.copy_(self.batchnorm[str(name)].weight.data)
                module.bias.data.copy_(self.batchnorm[str(name)].bias.data)
                module.running_var = self.batchnorm[str(name)].running_var
                module.running_mean = self.batchnorm[str(name)].running_mean

    def add_dataset(self, dataset, num_outputs): 
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(512, num_outputs, bias = True))

        initial_bn = nn.ModuleDict()
        for name, module in enumerate(self.shared.modules()):   

            if isinstance(module, nn.BatchNorm2d):
                new_bn = copy.deepcopy(module)
                new_bn.weight.data.fill_(1) 
                new_bn.bias.data.zero_()
                initial_bn[str(name)] = new_bn
        self.batchnorms.append(initial_bn)

    def save_bn(self):
        for name, module in enumerate(self.shared.modules()): # save bn value
            if isinstance(module, nn.BatchNorm2d):
                self.batchnorm[str(name)].weight.data.copy_(module.weight.data)
                self.batchnorm[str(name)].bias.data.copy_(module.bias.data)
                self.batchnorm[str(name)].running_var = module.running_var
                self.batchnorm[str(name)].running_mean = module.running_mean

    def forward(self, x):
        x = self.shared(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class Pruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, prune_perc, previous_masks, device = 'cuda:0'):
        self.model = model
        self.prune_perc = prune_perc

        self.train_bias = False
        self.train_bn = False
        self.current_masks = None
        self.previous_masks = previous_masks
        valid_key = list(previous_masks.keys())[-1] 
        self.current_dataset_idx = previous_masks[valid_key].max() 
        self.device = device

    # Modify mask based on pruning result
    def pruning_mask_weights(self, weights, previous_mask, layer_name):
        """Pruning criterion: Based on the fisher matrix. 
        """
        # Select all prunable weights, ie. belonging to current dataset.
        previous_mask = previous_mask.to(device)
        tensor = weights[previous_mask.eq(self.current_dataset_idx)] 
        abs_tensor = tensor.abs()
        cutoff_rank = round(self.prune_perc * tensor.numel()) 
        cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0]
        remove_mask = weights.abs().le(cutoff_value) * previous_mask.eq(self.current_dataset_idx)

        previous_mask[remove_mask.eq(1)] = 0 #set zero to pruned weights
        mask = previous_mask
        print('Layer #%s, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_name, mask.eq(0).sum(), tensor.numel(),
               100 * mask.eq(0).sum() / tensor.numel(), weights.numel()))
        return mask
    
    # Zero the pruning process and the weights being pruned
    def prune(self):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """
        print('Pruning for dataset idx: %d' % (self.current_dataset_idx))

        self.previous_masks = self.current_masks
        print('Pruning each layer by removing %.2f%% of values' % (100 * self.prune_perc))
        for n, p in self.model.shared.named_parameters():  
            n = n.replace('.', '__')
            if ('conv' in n) or ('shortcut__0__weight' in n):

                if p.requires_grad:
                    p_ = p.detach().clone()
                    mask = self.pruning_mask_weights(p_, self.previous_masks[n], n)
                    self.current_masks[n] = mask.to(self.device) 
                    p = p.detach()
                    p[self.current_masks[n].eq(0)] = 0.0  

    # Make the gradient zero other than the weight of the current task so that it does not become an update.
    def make_grads_zero(self):                      
        """Sets grads of fixed weights to 0."""
        assert self.current_masks
        for n, p in self.model.shared.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if ('conv' in n) or ('shortcut__0__weight' in n):
                    layer_mask = self.current_masks[n]
                    if p.grad is not None:
                        p.grad.data[layer_mask.ne(self.current_dataset_idx)] = 0 
                    

    # Make weight zero so that pruned weights are not involved in learning.
    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.current_masks
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('shortcut__0__weight' in n):
                layer_mask = self.current_masks[n]
                p_clone = p.detach()
                p_clone[layer_mask.eq(0)] = 0.0


    # In the forward step, only the weight of the previous tasks where the mask is not zero is used together.
    def apply_mask(self, dataset_idx):
        """To be done to retrieve weights just for a particular dataset."""
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('shortcut__0__weight' in n):
                mask = self.previous_masks[n].to(self.device)
                p_clone = p.detach() 
                p_clone[mask.eq(0)] = 0.0   
                p_clone[mask.gt(dataset_idx)] = 0.0  


    # For the first task, use only the weight of the task itself.
    def apply_mask_task_one(self, dataset_idx):
        """To be done to retrieve weights just for a particular dataset."""
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('shortcut__0__weight' in n):
                mask = self.previous_masks[n].to(self.device)
                p = p.detach()
                p[mask.ne(dataset_idx)] = 0.0 


    def initialize_first_mask(self):
        assert self.previous_masks
        self.current_masks = self.previous_masks 

    # Update the mask before learning the newly introduced task.
    def initialize_new_mask(self):
        """
        Turns previously pruned weights into trainable weights for
        current dataset.
        """
        assert self.previous_masks
        self.current_dataset_idx += 1
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('shortcut__0__weight' in n):
                mask = self.previous_masks[n]
                mask[mask.eq(0)] = self.current_dataset_idx 
        self.current_masks = self.previous_masks

    # Manage continuously so that the weight of the previous tasks does not change during learning.
    def concat_original_model(self, dataset_idx, original_model):
        for (n, p), (original_n, original_p) in zip(self.model.shared.named_parameters(), original_model.shared.named_parameters()):
            n = n.replace('.', '__')
            if ('conv' in n) or ('shortcut__0__weight' in n):
                weight = p.detach()
                original_weight = original_p.detach()
                mask = self.previous_masks[n].to(self.device)
                mask_ = mask.lt(dataset_idx).__and__(mask.gt(0))  
                weight[mask_.eq(1)] = original_weight[mask_.eq(1)]

    # Initializing weights for new tasks.
    def initialize_new_masked_weights(self, dataset_idx):
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('shortcut__0__weight' in n):
                weight = p.detach()
                mask = self.previous_masks[n].to(self.device)
                random_init = 0.01 * torch.randn((p.size())).to(self.device)
                weight[mask.eq(dataset_idx)] = random_init[mask.eq(dataset_idx)]

class Manager(object):
    """Handles training and pruning."""

    def __init__(self, model, pruning_rate, previous_masks, train_data, test_data,  dataset_list, device = 'cuda:0'):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.criterion = nn.CrossEntropyLoss()
        self.pruning_rate = pruning_rate
        self.device = device
        self.dataset_list = dataset_list
        self.pruner = Pruner(self.model, self.pruning_rate, previous_masks, device = self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones = [50, 75], gamma = 0.1)

    def eval(self, dataset_idx, biases=None):
        """Performs evaluation."""

        self.model = self.model.to(self.device)

        if dataset_idx == 1:
            print('Apply Mask: Dataset idx :', dataset_idx)
            self.pruner.apply_mask_task_one(dataset_idx)
        else:
            print('Apply Mask :', dataset_idx)
            self.pruner.apply_mask(dataset_idx)

        self.model.eval()

        test_loss = 0
        total = 0
        correct = 0

        print('Performing eval...')
        with torch.no_grad():
            for step, (image, label) in enumerate(self.test_data):

                image = image.to(self.device)
                label = label.to(self.device)
                image, label = Variable(image), Variable(label)

                pred = self.model(image)
                loss = self.criterion(pred, label)
                test_loss += loss.item()

                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0) 
        
        acc = 100 * correct / total
        print('[test] loss: {:.3f} | acc: {:.3f}'.format(test_loss/(step+1), acc))

        return acc, test_loss/(step+1)


    def train(self, dataset_idx, epochs, isRetrain=False):
        self.model = self.model.to(self.device)

        if isRetrain == 0 :
            self.optimizer =  optim.SGD(self.model.parameters(), lr=0.01)
        elif isRetrain == 1: 
            self.optimizer = optim.SGD(self.model.parameters(), lr = 0.001)

        if dataset_idx > 1: 
            original_model = copy.deepcopy(self.model)

        # In training step, initialize the weights that set to zero
        if isRetrain == 0 and dataset_idx > 1:               
            self.pruner.initialize_new_masked_weights(dataset_idx)            

        for epoch in range(epochs):
            epoch_idx = epoch + 1
            running_loss = 0
            acc = 0
            total = 0
            correct = 0
            best_acc = -1
            optimizer = self.optimizer

            """Apply mask after pruning"""
            if isRetrain == 1 and dataset_idx > 1: 
                self.pruner.apply_mask(dataset_idx)
            elif isRetrain == 1 and dataset_idx == 1: 
                self.pruner.apply_mask_task_one(dataset_idx)

            """Apply mask while training""" 
            self.pruner.apply_mask(dataset_idx)

            for step, (image, label) in enumerate(self.train_data):
                
                image = image.to(self.device)
                label = label.to(self.device)
                image, label = Variable(image), Variable(label)

                self.model.zero_grad()

                pred = self.model(image)
                loss = self.criterion(pred, label)
                
                loss.backward()

                self.pruner.make_grads_zero()   
                optimizer.step()                
                self.pruner.make_pruned_zero()  
                running_loss += loss.item()

                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)
            self.lr_scheduler.step()
            acc = 100 * (correct / total)
            print('train epoch : {} | loss: {:.3f} | acc: {:.3f}'.format(epoch_idx, running_loss/(step+1), acc))
            print('==============='* 4)

            if dataset_idx > 1:
                self.pruner.concat_original_model(dataset_idx, original_model)

        model = self.model
        ckpt = {
            'previous_masks': self.pruner.current_masks,
            'model': model,
        }

    def prune(self, dataset_idx, retrain_epochs):
        """Perform pruning."""

        self.pruner.prune()
        # retraining after pruning
        print('retraining after pruning...')
        self.train(dataset_idx, retrain_epochs, True)
        self.save_model(dataset_idx) #save final version of model

    def save_model(self, dataset_idx):
        """Save model to file."""
        model = self.model

        # Save Task specific BN 
        self.model.save_bn()
        ckpt = {
            'previous_masks': self.pruner.current_masks,
            'model': model,
        }
        torch.save(ckpt, 'packnet_' + self.dataset_list[dataset_idx-1] +'.pt')
        

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
    
    train_dataset_list = [CIFAR_100_train, MNIST_train, CIFAR_10_train]
    test_dataset_list = [CIFAR_100_test, MNIST_test, CIFAR_10_test]
    dataset_classes = [100, 10, 10]
    dataset_names = ['CIAFAR100', 'MNIST', 'CIFAR10']

    pruning_ratio = 0.75
    train_epochs = 100
    retrain_epochs = 10
    
    for task_idx in range(len(dataset_names)):
        model = ModifiedResNet18()
        model = model.to(device)
        previous_masks = {}

        for n, p in model.shared.named_parameters():
            if p.requires_grad: 
                n = n.replace('.', '__')
                if ('conv' in n) or ('shortcut__0__weight' in n):
                    mask = torch.randn(p.size()).fill_(1)
                    mask = mask.to(device)
                    previous_masks[n] = mask
        
        if task_idx > 0:
            ckpt = torch.load('packnet_' + dataset_names[task_idx-1] + '.pt', map_location = device)
            model = ckpt['model']
            previous_masks = ckpt['previous_masks']
        
        print('Training for {}'.format(dataset_names[task_idx]))
        
        train_dataset, test_dataset = train_dataset_list[task_idx], test_dataset_list[task_idx]
        model.add_dataset(dataset_names[task_idx], dataset_classes[task_idx])   
        model.set_dataset(dataset_names[task_idx])              
            
        manager = Manager(model, pruning_ratio, previous_masks, train_dataset, test_dataset, dataset_names, device= device)

        ## prune된 부분에 새로운 태스크의 parameter를 할당하는 부분 
        if task_idx == 0 : 
            manager.pruner.initialize_first_mask() 
        else: 
            manager.pruner.initialize_new_mask()

        manager.train(task_idx + 1, train_epochs)
        print('task {} pruning...'.format(dataset_names[task_idx]))
        manager.prune(task_idx+1, retrain_epochs)
        task_acc, loss = manager.eval(dataset_idx = task_idx + 1)
