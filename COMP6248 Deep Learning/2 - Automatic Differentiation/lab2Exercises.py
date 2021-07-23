import torch 
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

torch.manual_seed(0)

print("## PART 1 ##")
## Ex 1.1
def sgd_factorise_ad(A: torch.Tensor, rank: int, num_epochs = 1000, lr = 0.01):
    loss_list = np.empty((1,0))

    m = A.shape[0]
    n = A.shape[1]
    U = torch.rand(m, rank, dtype=A.dtype, requires_grad=True)
    V = torch.rand(n, rank, dtype=A.dtype, requires_grad=True)

    for epoch in range(num_epochs):
        #clear grads 
        U.grad = None
        V.grad = None

        #calc error and grad
        e = torch.nn.functional.mse_loss(A, U @ V.t(), reduction='sum')
        e.backward()

        #update using grad
        with torch.no_grad():
            U -= lr * U.grad
            V -= lr * V.grad
    return [U, V]

## Ex 1.2
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
data = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values)
data = data - data.mean(dim=0)

## own implementation
[U, V] = sgd_factorise_ad(data, 2)
R = U @ V.t()
loss = torch.nn.functional.mse_loss(R, data, reduction="sum")
print("Loss: ", loss.item())

## t-svd
U, S, V = torch.svd(data)
print(S)
S[2:] = 0
SVD = (U @ torch.diag(S)) @ V.t()
loss = torch.nn.functional.mse_loss(SVD, data, reduction="sum")
print("Loss: ", loss.item())

## Ex 1.3
figs, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
k = 2
U, S, V = torch.svd(data,some=False)
pcs = (U[:,:k] @ torch.diag(S)[:k,:k]) @ V[:,:k].t()
ax[0].scatter(pcs[:,0], pcs[:,1])
ax[0].set_title("Data projected on PC's")
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")

ax[1].scatter(U[:, 0], U[:, 1])
ax[1].set_title("First columns of U")
ax[1].set_xlabel("$U_{1}$")
ax[1].set_ylabel("$U_{2}$")

#figs.savefig("pcacomparison")
print()
print("## PART 2 ##")

import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = df.sample(frac=1) #shuffle
# add label indices column
mapping = {k : v for v , k in enumerate(df[4].unique())}
df[5] = df[4].map(mapping)
#normalise data
alldata = torch.tensor(df.iloc[:,[0,1,2,3]].values, dtype=torch.float)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)
#create datasets
targets_tr = torch.tensor(df.iloc[:100, 5].values, dtype=torch.long)
targets_va = torch.tensor(df.iloc[100:, 5].values, dtype=torch.long)
data_tr = alldata[:100]
data_va = alldata[100:]

## Ex 2.1

def MLP_train(data: torch.Tensor, targets: torch.Tensor, num_epochs = 100, lr = 0.01):
    W1 = torch.randn(4, 12, requires_grad=True)
    W2 = torch.randn(12, 3, requires_grad=True)
    b1 = torch.tensor(0.0, requires_grad=True)
    b2 = torch.tensor(0.0, requires_grad=True)

    for epoch in range(num_epochs):
        #reset grad
        W1.grad = None
        W2.grad = None
        b1.grad = None
        b2.grad = None

        #calc weight and grad
        logits = torch.relu(data @ W1 + b1) @ W2 + b2
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
       # print("Epoch ", epoch, " Loss: ", loss.item())

        #update using grad
        W1.data = W1 - lr * W1.grad
        W2.data = W2 - lr * W2.grad
        b1.data = b1 - lr * b1.grad
        b2.data = b2 - lr * b2.grad

    loss = torch.nn.functional.cross_entropy(logits, targets).item()
    return [[W1, b1], [W2,b2], loss]    

## Ex 2.2
[[W1, b1], [W2,b2], loss] = MLP_train(data_tr, targets_tr)
print("Training: ", loss)
val = torch.relu(data_va @ W1 + b1) @ W2 + b2
val_acc = torch.nn.functional.cross_entropy(val, targets_va)
print("Validation: " , val_acc.item())
print()

#trials = 1000
#tLoss = torch.empty(trials, dtype=torch.float)
#vLoss = torch.empty(trials, dtype=torch.float)
#print("Trials: ",trials)

#for i in range(trials):
#    [[W1, b1], [W2,b2], TLoss] = MLP_train(data_tr, targets_tr)
#    val = torch.relu(data_va @ W1 + b1) @ W2 + b2
#    val_acc = torch.nn.functional.cross_entropy(val, targets_va)
#    tLoss[i] = TLoss
#    vLoss[i] = val_acc.item()
#print("Training mean: ", torch.mean(tLoss), " Validation mean: ", torch.mean(vLoss))