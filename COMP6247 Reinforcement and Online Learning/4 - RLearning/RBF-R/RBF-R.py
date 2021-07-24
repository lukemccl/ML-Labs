import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_excel('data.xls',header=1,usecols='C:J')
cols = df.columns.tolist()
cols = cols[1:] + [cols[0]]
df = df[cols]

rawData = df.to_numpy()# Dataset of your chice
N, pp1 = rawData.shape
# Last column is target
X = np.matrix(rawData[:,0:pp1-1])
y = np.matrix(rawData[:,pp1-1]).T
print(X.shape, y.shape)

# Solve linear regression, plot target and prediction
w = (np.linalg.inv(X.T*X)) * X.T * y
yh_lin = X*w

# J = 20basis functions obtained by k-means clustering
# sigma set to standard deviation of entire data
from sklearn.cluster import KMeans
J = 20;
kmeans = KMeans(n_clusters=J, random_state=0).fit(X)
sig = np.std(X)
# Construct design matrix
U = np.zeros((N,J))
for i in range(N):
	for j in range(J):
		U[i][j] = np.exp(-np.linalg.norm(X[i] - kmeans.cluster_centers_[j])/sig)
		
# Solve RBF model, predict and plot
w = np.dot((np.linalg.inv(np.dot(U.T,U))), U.T) * y
yh_rbf = np.dot(U,w)
plt.figure(figsize=(6,6))
plt.plot(y, yh_lin, '.', Color='magenta', label='linear')
plt.plot(y, yh_rbf, '.', Color='cyan', label='RBF')
plt.title('Targets and predictions of Linear and RBF')
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.legend()
plt.savefig('CF.png')
plt.show()
print(np.linalg.norm(y-yh_lin), np.linalg.norm(y-yh_rbf))

T = 1
lr = 1e-2

w_mse = np.zeros(7)
for i in range(7):
	t0 = 10**i
	
	w_sgd = np.random.randn(J,1)
	loss = np.zeros(t0)

	for t in range(t0):
		t_idx = np.random.randint(0, N)
		U_t = U[t_idx:t_idx+1]
		y_t = y[t_idx:t_idx+1]

		pred_t = np.dot(U_t, w_sgd)
		err_t = y_t - pred_t

		w_sgd = w_sgd + lr * np.dot(U_t.T, err_t)
		loss[t] = np.linalg.norm(y - np.dot(U,w_sgd))
    
	plt.figure(figsize=(6,6))    
	plt.plot(loss)
	plt.title('Mean Squared Value Error \n (J ={})'.format(J))
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.savefig('LOSS-{}.png'.format(t0))
	plt.clf()

	yh_sgd = np.dot(U,w_sgd)
	plt.figure(figsize=(6,6))
	plt.plot(y, yh_rbf, '.', Color='magenta', label='RBF closed')
	plt.plot(y, yh_sgd, '.', Color='cyan', label='RBF SGD')
	plt.title('The targets and predictions of \n SGD closed form and RBF by using SGD')
	plt.xlabel('Targets')
	plt.ylabel('Predictions')
	plt.legend()
	plt.savefig('SGD-{}.png'.format(t0))
	plt.clf()
	
	w_mse[i] = np.linalg.norm(w-w_sgd)

plt.figure(figsize=(6,6))    
plt.plot(w_mse)
plt.title('Mean Squared Value Error of weights')
plt.xlabel('Epochs ($10^x$)')
plt.ylabel('Error')
plt.savefig('LWEIGHTS.png'.format(t0))
plt.clf()