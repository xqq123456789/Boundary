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
import time
from pyheatmap.heatmap import HeatMap
import pandas as pd
import seaborn as sns

start = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)  
EPOCH = 1 
BATCH_SIZE = 1
LR = 0.001
DOWNLOAD_MNIST = False

# 下载mnist手写数据集
train_data = torchvision.datasets.FashionMNIST(
    root='./F_MNIST_data/', 
    train=True, 
    transform=torchvision.transforms.ToTensor(), 
    download=True, 
)

test_data = torchvision.datasets.FashionMNIST(
    root='./F_MNIST_data/',
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


#求输入x与梯度w的乘积
def multi_x_w(x,w):
	multi=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],dtype=float)
	for i in range(0,10):
		sum_num=0
		for j in range(0,28):
			for k in range(0,28):
				sum_num+=x[0][0][j][k]*w[0][i][j][k]
		multi[0][i]=sum_num
	return multi

#求梯度
def gradient(out,s_i):	
	for i in range(0,10):
		x=torch.tensor([[0,0,0,0,0,0,0,0,0,0]],dtype=float).to(device)
		x[0][i]=1
		out.backward(gradient=x,retain_graph=True)
		temp=copy.deepcopy(s_i.grad)
		s_i.grad.data.zero_()
		if i>0:
			temp=torch.cat((temp1,temp),1)
		temp1=temp
	#print(temp.size())
	return temp

#二分搜索
def binary_search(x_pos,x_neg):
	while True:
		s_i=(r*x_pos+(1-r)*x_neg).to(device)
		fs_i,v1,v2=cnn(s_i)
		#pred_y= torch.max(fs_i, 1)[1].data.numpy()#预测label
		pred_y= torch.max(fs_i, 1)[1].data.cpu().numpy()#预测label
		result=np.sort(fs_i.detach().cpu().numpy()[0])
		max1=result[-1]
		max2=result[-2]
		#print(label,pred_y,result)
		if pred_y[0]==x_s_label and max1-max2<e:
			break;
		if pred_y[0]==x_s_label:
			x_pos=s_i
		else:
			x_neg=s_i
	s_i=Variable(s_i,requires_grad=True)
	return 	s_i

#在train_data中选择一个固定标签的样本作为seed_data
def seed_data(train_data,x_s_label):
	for x in enumerate(train_data):	
		if x[1][1]==x_s_label:
			x_s=x[1][0]
	return x_s
		
#求边界,H包括s_i的类别及w,b
def one_boundary(out,s_i,boundary,label):
	w=gradient(out,s_i).to(device)
	multi=multi_x_w(s_i,w).to(device)
	b=out-multi
	H=[label,w,b]
	boundary.append(H)
	return boundary	

#求热力图
def heatmap(R,num):
	fig = plt.figure()
	sns_plot = sns.heatmap(R)
	fig.savefig("heatmap"+str(num)+".png", bbox_inches='tight') # 减少边缘空白

#求boundary的质量
def quality(w,b,x_s_label,target_label,data):
	total_num=0
	true_num=0
	cnn_num=0
	for label,x_i in enumerate(data):
		if test_y[label].item()==x_s_label or test_y[label].item()==target_label:
			total_num+=1
			x_i=torch.unsqueeze(x_i, dim=1).type(torch.FloatTensor).to(device)
			mul_r=multi_x_w(x_i,w).to(device)
			result=mul_r+b
			out,v1,v2=cnn(x_i)
			pred_y= torch.max(out, 1)[1].data.cpu().numpy()
			#print(test_y[label],out,result[0][0]>result[0][7])
			if (test_y[label].item()==x_s_label and result[0][x_s_label]>result[0][target_label]) or (test_y[label].item()==target_label and result[0][x_s_label]<result[0][target_label]):
				true_num+=1
			if test_y[label].item()==pred_y:
				cnn_num+=1
	return total_num,true_num,cnn_num

#求seed是否在里面
def seed_quality(w,b,x_s_label,x_s,target_label):
	x_s=torch.unsqueeze(x_s, dim=1).type(torch.FloatTensor).to(device)
	mul_r=multi_x_w(x_s,w).to(device)
	result=mul_r+b
	out,v1,v2=cnn(x_s)
	pred_y= torch.max(out, 1)[1].data.cpu().numpy()
	if result[0][x_s_label]>result[0][target_label]:
		return 1
	else:
		return 0

#对每一个seed，求boundary
def total_boundary(x_s_label,x_s):
	boundary=[]
	for label,x_i in enumerate(test_x[0:20]):
		if test_y[label].item()!=x_s_label:
			x_pos=torch.unsqueeze(x_s, dim=1).type(torch.FloatTensor).to(device)
			x_neg=torch.unsqueeze(x_i, dim=1).type(torch.FloatTensor).to(device)
			s_i=binary_search(x_pos,x_neg)
			out,v1,v2=cnn(s_i)
			boundary=one_boundary(out,s_i,boundary,test_y[label].item())
	return boundary
		
cnn = CNN()
if torch.cuda.is_available():
    cnn.cuda()
cnn.load_state_dict(torch.load('cnn2_fminist.pkl'))
cnn.eval()
r=0.9
e=0.1

for i in range(20):	
	x_s_label=i #设置seed_data的label
	x_s=seed_data(train_data,x_s_label)
	boundary=total_boundary(x_s_label,x_s)
	print(len(boundary),len(boundary[0]))
	for j in range(len(boundary)):
		target_label=boundary[j][0]
		print(target_label)
		w=boundary[j][1]
		b=boundary[j][2]

		'''for i in range(10):
			R=w[0][i].cpu().numpy().tolist()
			heatmap(R,i)
		R=(w[0][target_label]-w[0][0]).cpu().numpy().tolist()
		heatmap(R,11)'''
		print(seed_quality(w,b,x_s_label,x_s,target_label)) #seed判断是否正确
		print(quality(w,b,x_s_label,target_label,test_x[0:50]))　#boundary的正确率	

end = time.time()
print(end-start)

