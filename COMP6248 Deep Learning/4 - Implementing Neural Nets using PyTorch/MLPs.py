import torch
import torchbearer
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
device = "cuda:0" if torch.cuda.is_available() else "cpu"

## TODO: NEED TO RERUN TO REGENERATE LEARNINGCURVES

# fix random seed for reproducibility
seed = 7
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(seed)

# flatten 28*28 images to a 784 vector for each image
transform = transforms.Compose([
    transforms.ToTensor(),  # convert to tensor
    transforms.Lambda(lambda x: x.view(-1))  # flatten into vector
])

# load data
trainset = MNIST(".", train=True, download=True, transform=transform)
testset = MNIST(".", train=False, download=True, transform=transform)

# create data loaders
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=True)

# define baseline model
class BaselineModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(BaselineModel, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size) 
		self.fc2 = nn.Linear(hidden_size, num_classes)  
    
	def forward(self, x):
		out = self.fc1(x)
		out = F.relu(out)
		out = self.fc2(out)
		if not self.training:
			out = F.softmax(out, dim=1)
		return out

loss_fn = nn.CrossEntropyLoss()

N = 19
lowN = 0
epochs = 25

modelsAcc = np.empty((N-lowN, 2))
modelsLoss = np.empty((N-lowN, 2))

if(True):
    learningcurves = np.empty((N-lowN, 2, epochs))

    for i in range(lowN, N):
        print('Hidden units: ', 2**i)
        model = BaselineModel(784, 2**i, 10).to(device)
        optimiser = optim.Adam(model.parameters())

        trial = torchbearer.Trial(model, optimiser, loss_fn, metrics=['loss','accuracy']).to(device)
        trial.with_generators(trainloader, test_generator=testloader)
        state = trial.run(epochs=epochs)
        ##measure test loss and accuracy per epoch
    
        losses = np.asarray([state[j].get('loss') for j in range(epochs)])
        accs = np.asarray([state[j].get('acc') for j in range(epochs)])
        testlosses = np.asarray([state[j].get('test_loss') for j in range(epochs)])
        testaccs = np.asarray([state[j].get('test_acc') for j in range(epochs)])
        results = trial.evaluate(data_key=torchbearer.TEST_DATA)

        print('Hidden units: ', 2**i, ' Results: ', results)

        modelsAcc[i-lowN] = np.asarray([state[-1].get('acc'), results.get('test_acc')])
        modelsLoss[i-lowN] = np.asarray([state[-1].get('loss'), results.get('test_loss')])

        learningcurves[i-lowN, 0] = losses
        learningcurves[i-lowN, 1] = accs

    np.savetxt("modelsAcc.csv", modelsAcc, delimiter=",")
    np.savetxt("modelsLoss.csv", modelsLoss, delimiter=",")

else:
    modelsAcc = np.genfromtxt("modelsAcc.csv", delimiter=",")
    modelsLoss = np.genfromtxt("modelsLoss.csv", delimiter=",")

xs = [2**n for n in range(lowN, N)]

plt.plot(xs, modelsAcc[:,0], label="Train Acc",linestyle='--', c='r')
plt.plot(xs, modelsAcc[:,1], label="Test Acc",linestyle='-', c='r')
plt.plot(xs, modelsLoss[:,0], label="Train Loss",linestyle='--', c='b')
plt.plot(xs, modelsLoss[:,1], label="Test Loss",linestyle='-', c='b')
plt.xlabel("Num hidden units")
plt.legend()
plt.xscale("log")
plt.savefig("persizelog.png")
plt.show()

plt.plot(xs, modelsAcc[:,0], label="Train Acc",linestyle='--', c='r')
plt.plot(xs, modelsAcc[:,1], label="Test Acc",linestyle='-', c='r')
plt.plot(xs, modelsLoss[:,0], label="Train Loss",linestyle='--', c='b')
plt.plot(xs, modelsLoss[:,1], label="Test Loss",linestyle='-', c='b')
plt.xlabel("Num hidden units")
plt.legend()
plt.savefig("persizelin.png")
plt.show()

plt.plot(xs, modelsAcc[:,0], label="Train Acc",linestyle='--', c='r')
plt.plot(xs, modelsAcc[:,1], label="Test Acc",linestyle='-', c='r')
plt.xlabel("Num hidden units")
plt.legend()
plt.xscale("log")
plt.savefig("acclog.png")
plt.show()

plt.plot(xs, modelsAcc[:,0], label="Train Acc",linestyle='--', c='r')
plt.plot(xs, modelsAcc[:,1], label="Test Acc",linestyle='-', c='r')
plt.xlabel("Num hidden units")
plt.legend()
plt.savefig("acclin.png")
plt.show()

plt.plot(xs, modelsLoss[:,0], label="Train Loss",linestyle='--', c='b')
plt.plot(xs, modelsLoss[:,1], label="Test Loss",linestyle='-', c='b')
plt.xlabel("Num hidden units")
plt.legend()
plt.xscale("log")
plt.savefig("losslog.png")
plt.show()

for i in range(len(learningcurves)):
    legend = str(2**(lowN+i))
    plt.plot(learningcurves[i,0,:], label=legend)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("allLosses.png")
plt.show()

for i in range(len(learningcurves)):
    legend = str(2**(lowN+i))
    plt.plot(learningcurves[i,1,:], label=legend)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')    
plt.legend()
plt.savefig("allacc.png")   
plt.show()