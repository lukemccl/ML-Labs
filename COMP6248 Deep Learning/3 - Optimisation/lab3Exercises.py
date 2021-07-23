import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

print('## Task 1 ##')

def rastrigin(x, A=1):
    sx = 0
    for xi in x:
        sx += xi**2 - A*torch.cos(2*np.pi*xi)
    return A*x.shape[0] + sx
    
p = torch.tensor([5.0,5.0], requires_grad=True)
optSGD = optim.SGD([p], lr=0.01)
loss_SGD = np.empty((1,0))

#q = torch.tensor([5.0,5.0], requires_grad=True)
#optSGDM = optim.SGD([q], lr=0.01, momentum=0.9)
#loss_SGDM = np.empty((1,0))

#r = torch.tensor([5.0,5.0], requires_grad=True)
#optADAG = optim.Adagrad([r], lr=0.01)
#loss_ADAG = np.empty((1,0))

#s = torch.tensor([5.0,5.0], requires_grad=True)
#optADAM = optim.Adam([s], lr=0.01)
#loss_ADAM = np.empty((1,0))

for i in range(0):
    optSGD.zero_grad()
    optSGDM.zero_grad()
    optADAG.zero_grad()
    optADAM.zero_grad()
    
    outputSGD = rastrigin(p)
    outputSGDM = rastrigin(q)
    outputADAG = rastrigin(r)
    outputADAM = rastrigin(s)

    outputSGD.backward()
    outputSGDM.backward()
    outputADAG.backward()
    outputADAM.backward()
    
    optSGD.step()
    optSGDM.step()
    optADAG.step()
    optADAM.step()
    
    loss_SGD  = np.append(loss_SGD, outputSGD.detach())
    loss_SGDM = np.append(loss_SGDM, outputSGDM.detach())
    loss_ADAG = np.append(loss_ADAG, outputADAG.detach())
    loss_ADAM = np.append(loss_ADAM, outputADAM.detach())

#print("SGD: ", p)
#print("SGDM: ", q)
#print("ADAG: ", r)
#print("ADAM: ", s)
    
#plt.plot(loss_SGD, color='red', label='SGD')
#plt.plot(loss_SGDM, color='black', label='SGDM')
#plt.plot(loss_ADAG, color='blue', label='ADAG')
#plt.plot(loss_ADAM, color='green', label='ADAM')
#plt.xlabel("Epoch")
#plt.ylabel("Rastrigin(x)")#
#plt.legend()
#plt.savefig("optimisers")

#plt.show()

print()
print('## Task 2 ##')
import torch
from torch.utils import data
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = df.sample(frac=1, random_state=0) #shuffle

df = df[df[4].isin(['Iris-virginica','Iris-versicolor'])] #filter

#add label indices column
mapping = {k: v for v, k in enumerate(df[4].unique())}
df[5] = (2 * df[4].map(mapping)) -1 #labels in{-1,1}

#normalise data
alldata = torch.tensor(df.iloc[:,[0,1,2,3]].values, dtype=torch.float)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)

#create datasets
targets_tr = torch.tensor(df.iloc[:75, 5].values, dtype=torch.long)
targets_va = torch.tensor(df.iloc[75:, 5].values, dtype=torch.long)
data_tr = alldata[:75]
data_va = alldata[75:]

def hinge_loss(y_pred, y_true):
    return torch.mean(torch.max(torch.zeros(y_pred.shape),1-y_pred*y_true))

def svm(x, w, b):
    h = (w*x).sum(1) + b
    return h
    
tr_dataset = data.TensorDataset(data_tr,targets_tr) # create your datset
va_dataset = data.TensorDataset(data_va,targets_va)
tr_loader = data.DataLoader(tr_dataset, batch_size=25, shuffle=True) # create your dataloader
va_loader = data.DataLoader(va_dataset, batch_size=25, shuffle=True) # create your dataloader


def SVMOPT(train, validate, l_r=0.01, w_decay=0.0001, epochs=100, sgd=True, acc=False):
    w = torch.randn(1, 4, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    opt = optim.SGD([w,b], lr=l_r, weight_decay=w_decay) if sgd else optim.Adam([w,b], lr=l_r, weight_decay=w_decay)

    for epoch in range(epochs):
        for batch in train:
            opt.zero_grad()

            output = svm(batch[0], w, b)
            loss = hinge_loss(output, batch[1])
            loss.backward()
        
            opt.step()
            
    tLoss = 0
    Acc = 0
    with torch.no_grad():
        output = svm(data_va, w, b)
        loss = hinge_loss(output, targets_va)
        tLoss += loss
        Acc = np.count_nonzero(output*targets_va > 0)/output.shape[0]
        if(acc):
            print(targets_va)
            print(output)
    if(acc):
        return Acc
    else:
        return loss
    
vAccsgd = SVMOPT(tr_loader, va_loader, acc=True)
vAccadm = SVMOPT(tr_loader, va_loader, sgd=False, acc=True)
print("SGD: ", vAccsgd)
print("ADAM: ", vAccadm)

samples = 200
sgdLosses = np.empty((1,0))
admLosses = np.empty((1,0))
for s in range(samples):
    sgdLosses = np.append(sgdLosses, SVMOPT(tr_loader, va_loader))
    admLosses = np.append(admLosses, SVMOPT(tr_loader, va_loader, sgd=False))
    
print("SGD: ", np.mean(sgdLosses))
print("ADAM: ", np.mean(admLosses))

def cum_mean(arr):
    cum_sum = np.cumsum(arr, axis=0)    
    for i in range(cum_sum.shape[0]):            
        cum_sum[i] =  cum_sum[i] / (i+1)
    return cum_sum
    
#plt.plot(cum_mean(sgdLosses), color='red', label='SGD')
#plt.plot(cum_mean(admLosses), color='blue', label='ADAM')
#plt.xlabel("Sample")
#plt.ylabel("Mean Loss")
#plt.legend()
#plt.savefig("svm")