import torch
from torch import nn
import torch.nn.functional as F
import vgg_cifar10
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vgg_cifar10.VGG('VGG16')
model.load_state_dict(torch.load('./model/vgg_cifar10.pt'))
model(torch.randn(1,3,32,32)).shape
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(32,32)),
    #transforms.CenterCrop(size=(224,224)),
])

with open("./data/cifar10_labels.txt") as f:
    idx2label = eval(f.read())

X = plt.imread("./data/maltipoo.jpg")
print(type(X))
print(X.shape)
plt.imshow(X)

X=transform(X)
print(type(X))
print(X.shape)
plt.figure()
plt.imshow(X.permute(1,2,0))

X = X.unsqueeze(dim=0).to(device)
print(X.shape)

model.eval()
with torch.no_grad():
    y_hat_sorted = model(X).sort(descending=True)
    percent_top5 = F.softmax(y_hat_sorted[0], dim=1) [0,:5]*100
    pred_top5 = y_hat_sorted[1][0,:5]
for i, idx in enumerate(pred_top5):
    print(idx2label[idx.item()], end="")
    print(f"({round(percent_top5[i].item(),1)} %)")
