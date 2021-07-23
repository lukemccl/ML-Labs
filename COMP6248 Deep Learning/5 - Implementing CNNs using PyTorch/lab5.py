import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchbearer
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchbearer import Trial
import torchbearer
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class MyDataset(Dataset):
  def __init__(self, size=5000, dim=40, random_offset=0):
        super(MyDataset, self).__init__()
        self.size = size
        self.dim = dim
        self.random_offset = random_offset

  def __getitem__(self, index):
      if index >= len(self):
          raise IndexError("{} index out of range".format(self.__class__.__name__))

      rng_state = torch.get_rng_state()
      torch.manual_seed(index + self.random_offset)

      while True:
        img = torch.zeros(self.dim, self.dim)
        dx = torch.randint(-10,10,(1,),dtype=torch.float)
        dy = torch.randint(-10,10,(1,),dtype=torch.float)
        c = torch.randint(-20,20,(1,), dtype=torch.float)

        params = torch.cat((dy/dx, c))
        xy = torch.randint(0,img.shape[1], (20, 2), dtype=torch.float)
        xy[:,1] = xy[:,0] * params[0] + params[1]

        xy.round_()
        xy = xy[ xy[:,1] > 0 ]
        xy = xy[ xy[:,1] < self.dim ]
        xy = xy[ xy[:,0] < self.dim ]

        for i in range(xy.shape[0]):
          x, y = xy[i][0], self.dim - xy[i][1]
          img[int(y), int(x)]=1
        if img.sum() > 2:
          break

      torch.set_rng_state(rng_state)
      return img.unsqueeze(0), params

  def __len__(self):
      return self.size

class BaselineCNN(nn.Module):
	def __init__(self):
		super(BaselineCNN, self).__init__()
		self.C1 = nn.Conv2d(1, 48, kernel_size=(3,3), stride=1, padding=1) 
		self.fc1 = nn.Linear(48 * 40**2, 128)
		self.fc2 = nn.Linear(128, 2)
		
	def forward(self, x):
		out = self.C1(x)
		out = F.relu(out)
		out = out.view(out.shape[0], -1)
		out = self.fc1(out)
		out = F.relu(out)
		out = self.fc2(out)
		return out
		
class ImprovedCNN(nn.Module):
	def __init__(self):
		super(ImprovedCNN, self).__init__()
		self.C1 = nn.Conv2d(1, 48, kernel_size=(3,3), stride=1, padding=1) 
		self.C2 = nn.Conv2d(48, 48, kernel_size=(3,3), stride=1, padding=1) 
		self.MP = nn.AdaptiveMaxPool2d((1, 1))
		self.fc1 = nn.Linear(48, 128)
		self.fc2 = nn.Linear(128, 2)
		
	def forward(self, x):
		out = self.C1(x)
		out = F.relu(out)
		out = self.C2(out)
		out = F.relu(out)
		out = self.MP(out)
		out = out.view(out.shape[0], -1)
		out = self.fc1(out)
		out = F.relu(out)
		out = self.fc2(out)
		return out		
		
class BestCNN(nn.Module):
	def __init__(self):
		super(BestCNN, self).__init__()
		self.C1 = nn.Conv2d(3, 48, kernel_size=(3,3), stride=1, padding=1) 
		self.C2 = nn.Conv2d(48, 48, kernel_size=(3,3), stride=1, padding=1) 
		self.MP = nn.AdaptiveMaxPool2d((1, 1))
		self.fc1 = nn.Linear(48, 128)
		self.fc2 = nn.Linear(128, 2)
		
	def forward(self, x):
		idxx = torch.repeat_interleave(
			torch.arange(-20, 20, dtype=torch.float).unsqueeze(0)/40.0,
			repeats=40, dim=0).to(x.device)
		idxy = idxx.clone().t()
		idx = torch.stack([idxx, idxy]).unsqueeze(0)
		idx = torch.repeat_interleave(idx, repeats=x.shape[0], dim=0)
		x = torch.cat([x ,idx], dim=1)
		
		out = self.C1(x)
		out = F.relu(out)
		out = self.C2(out)
		out = F.relu(out)
		out = self.MP(out)
		out = out.view(out.shape[0], -1)
		out = self.fc1(out)
		out = F.relu(out)
		out = self.fc2(out)
		exit()
		return out		

train_data = MyDataset()
val_data = MyDataset(size=500, random_offset=33333)
test_data = MyDataset(size=500, random_offset=99999)

trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
testloader = DataLoader(test_data, batch_size=128, shuffle=True)
valloader = DataLoader(val_data, batch_size=128, shuffle=True)

def trainAndEval(model, epochs, figname):
	# define the loss function and the optimiser
	loss_function = nn.MSELoss()
	optimiser = optim.Adam(model.parameters())

	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	trial = Trial(model, optimiser, loss_function, metrics=['loss', 'mse']).to(device)
	trial.with_generators(trainloader, val_generator=valloader, test_generator=testloader)
	state = trial.run(epochs=epochs)
	results = trial.evaluate(data_key=torchbearer.TEST_DATA)
	print(results)

	losses = np.asarray([state[j].get('loss') for j in range(epochs)])
	vlosses = np.asarray([state[j].get('val_loss') for j in range(epochs)])

	fig, ax = plt.subplots()
	ax.plot(losses, label="Train", c='b')
	ax.plot(vlosses, label="Val", c='r')
	ax.set_xlabel('Epochs')
	ax.set_ylabel('Loss')
	ax.legend()

	fig.savefig(str(figname+".png"))
	fig.show()
	
	return (losses, vlosses, results)

#(l1, vl1, r1) = trainAndEval(BaselineCNN(), 100, "CNN1") 
#(l2, vl2, r2) = trainAndEval(ImprovedCNN(), 100, "CNN2") 
(l3, vl3, r3) = trainAndEval(BestCNN(), 100, "CNN3") 


fig, ax = plt.subplots()
ax.plot(l1, label="CNN 1",linestyle='-', c='b')
ax.plot(vl1, linestyle='--', c='b')
ax.plot(l2, label="CNN 2",linestyle='-', c='r')
ax.plot(vl2, linestyle='--', c='r')
ax.plot(l3, label="CNN 3",linestyle='-', c='g')
ax.plot(vl3, linestyle='--', c='g')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()

fig.savefig(str("ALL.png"))
fig.show()
	
#print(r1)
#print(r2)
print(r3)