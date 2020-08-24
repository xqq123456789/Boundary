import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from torch.autograd import Variable
import copy

torch.manual_seed(1)  
EPOCH = 1 
BATCH_SIZE = 1
LR = 0.001
DOWNLOAD_MNIST = False

# 下载mnist手写数据集
train_data = torchvision.datasets.MNIST(
    root='./data/', 
    train=True, 
    transform=torchvision.transforms.ToTensor(), 
    download=DOWNLOAD_MNIST, 
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False 
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.targets[:2000]

# 用class类来建立CNN模型
grads = {}
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  
                in_channels=1,  
                out_channels=16, 
                kernel_size=5, 
                stride=1,  
                padding=2,  
            ),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), 
            # 输出图像大小(16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d( 
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类
    def save_grad(name):
        def hook(grad):
        	grads[name] = grad
        return hook
    def forward(self, x):
        v1 = self.conv1(x)  
        v1.retain_grad()
        v2 = self.conv2(v1)   
        v2.retain_grad() 
        v3 = v2.view(v2.size(0), -1) 
        output = self.out(v3)
        return output,v1,v2

cnn = CNN()
cnn.load_state_dict(torch.load('cnn2.pkl'))
cnn.eval()

#随意选择x_s
for x in enumerate(train_data):	
	if x[1][1]==0:
		x_s=x[1][0]
r=0.9
e=0.1
boundary=[]
for label,x_i in enumerate(test_x[0:100]):
	if test_y[label].item()>0:
		x_pos=torch.unsqueeze(x_s, dim=1).type(torch.FloatTensor)
		x_neg=torch.unsqueeze(x_i, dim=1).type(torch.FloatTensor)
		while True:
			s_i=r*x_pos+(1-r)*x_neg
			fs_i,v1,v2=cnn(s_i)
			pred_y= torch.max(fs_i, 1)[1].data.numpy()#预测label
			result=np.sort(fs_i.detach().numpy()[0])
			max1=result[-1]
			max2=result[-2]
			#print(label,pred_y,result)
			if pred_y[0]==0 and max1-max2<e:
				break;
			if pred_y[0]==0:
				x_pos=s_i
			else:
				x_neg=s_i
		s_i=Variable(s_i,requires_grad=True)
		out,v1,v2=cnn(s_i)
		#求梯度W
		for i in range(0,10):
			x=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],dtype=float)
			x[0][i]=1
			out.backward(gradient=x,retain_graph=True)
			temp=copy.deepcopy(s_i.grad)
			s_i.grad.zero_()
			if i>0:
				temp=torch.cat((temp1,temp),1)
			temp1=temp
			#print(out,temp1.size())

		b=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],dtype=float)
		for i in range(0,10):
			sum=0
			for j in range(0,28):
				for k in range(0,28):
					sum+=s_i[0][0][j][k]*temp[0][i][j][k]
			b[0][i]=sum
		x=[temp,b]
		boundary.append(x)
		#print(len(boundary))


