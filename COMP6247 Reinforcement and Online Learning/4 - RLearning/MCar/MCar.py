import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import gym
import numpy as np
env_name = "MountainCar-v0"
env = gym.make(env_name)
obs = env.reset()

# Some initializations
#
n_states = 40
episodes = 20
initial_lr = 1.0
min_lr = 0.005
gamma = 0.99
max_stps = 300
epsilon = 0.05
env = env.unwrapped
env.seed()
np.random.seed(0)

pos = np.linspace(-1.2, 0.6, n_states)
vel = np.linspace(-0.07, 0.07, n_states)
pos_grid, vel_grid = np.meshgrid(pos, vel)
actions = 3

# Display functions
#
def pSpace(space,s="Q table",zlabel="-Q value",filename="QTableRep",zlim=None):
	fig = plt.figure(figsize=(10,10))
	ax = Axes3D(fig)
	if(zlim is not None):
		ax.set_zlim3d(0, zlim)
	episode_max_action = np.min(space, 2)
	surf = ax.plot_surface(pos_grid, vel_grid, -episode_max_action, cstride=1, rstride=1, cmap=cm.jet)
	ax.set_title(s, fontsize=14)
	ax.set_xlabel('Position', fontsize=14)
	ax.set_ylabel('Velocity', fontsize=14)
	ax.set_zlabel(zlabel, fontsize=14)
	ax.view_init(30,-30)
	plt.savefig(filename)
	plt.close(fig)
	
def anim(q_hist, basename):
	i = 0
	zlim = -np.amin(q_hist)
	
	for e in q_hist:
		pSpace(e,s="",filename='anim/{}_e_{}'.format(basename, i),zlim=zlim)
		i+=1

def SARSA_anim(w_hist, c, s, basename):
	N = n_states*n_states
	action_grid = (0, env.action_space.n, env.action_space.n)
	states = np.dstack((pos_grid, vel_grid)).reshape(N, 2)
	X = np.append(np.tile(states, (3,1)), np.tile(action_grid, 1600).reshape(-1, 1), axis=1)

	def QFunc(state, w):
		#perform rbf
		U = np.exp(-np.linalg.norm(state - c, axis=1)/s)**2
		q = np.dot(U, w)
		return q
	
	q_hist = np.zeros((w_hist.shape[0], n_states, n_states, actions))

	for i in range(w_hist.shape[0]):
		w = w_hist[i]		
		acs = np.asarray([QFunc(state,w) for state in states])
		print(acs.shape)
		acs = np.squeeze(acs)

		q_hist[i] = acs.reshape(n_states, n_states, actions)
			
	i = 0
	zlim = -np.amin(q_hist)
	
	for e in q_hist:
		pSpace(e,s="",filename='SARSAanim/{}_e_{}'.format(basename, i),zlim=zlim)
		i+=1
		
# Quantize the states
#
def discretization(env, obs):
	env_low = env.observation_space.low
	env_high = env.observation_space.high
	env_den = (env_high - env_low) / n_states
	pos_den = env_den[0]
	vel_den = env_den[1]
	pos_high = env_high[0]
	pos_low = env_low[0]
	vel_high = env_high[1]
	vel_low = env_low[1]
	pos_scaled = int((obs[0] - pos_low) / pos_den)
	vel_scaled = int((obs[1] - vel_low) / vel_den)
	return pos_scaled, vel_scaled

def tabQuantize(episodes=20, display=False):
	obs = env.reset()
	q_table = np.zeros((n_states, n_states, env.action_space.n))
	q_hist = np.zeros((episodes, n_states, n_states, env.action_space.n))
	total_steps = 0
	n_steps = np.zeros((episodes))
	
	for episode in range(episodes):
		obs = env.reset()
		total_reward = 0
		alpha = max(min_lr, initial_lr*(gamma**(episode//100)))
		steps = 0
		while True:
			pos, vel = discretization(env, obs)
			if np.random.uniform(low=0, high=1) < epsilon:
				a = np.random.choice(env.action_space.n)
			else:
				a = np.argmax(q_table[pos][vel])
			obs, reward, terminate,_ = env.step(a)
			total_reward += abs(obs[0]+0.5)
			pos_, vel_ = discretization(env, obs)

			# Q function update
			#
			q_table[pos][vel][a] = (1-alpha)*q_table[pos][vel][a] + alpha*(reward+gamma*np.max(q_table[pos_][vel_]))
			steps += 1
			if terminate:
				n_steps[episode] = steps
				print("Episode:", episode, "Steps:",steps)
				break
		q_hist[episode] = q_table
	if display:
		fig=plt.figure(figsize=(6,6))
		ax = plt.axes()
		ax.plot(n_steps)
		ax.set_ylabel("Steps")
		ax.set_xlabel("Episode")
		fig.savefig('STEPS')
		plt.close(fig)
		
		anim(q_hist, 'Q_table')
	return q_table, q_hist
	
def RBF_build(J=20, q_table=None):
	from sklearn.cluster import KMeans

	if (q_table is None):
		q_table, q_hist = tabQuantize(1000)
	
	N = n_states*n_states
	X = np.dstack((pos_grid, vel_grid)).reshape(N, 2)
	Y = q_table.reshape(n_states*n_states, actions)
	sig = np.std(X)
	kmeans = KMeans(n_clusters=J, random_state=0).fit(X)
	
	# Construct design matrix
	U = np.zeros((N,J))
	for i in range(N):
		for j in range(J):
			U[i][j] = np.exp(-np.linalg.norm(X[i] - kmeans.cluster_centers_[j])/sig)**2

	w = np.dot((np.linalg.inv(np.dot(U.T,U))), U.T) @ Y
	y = np.dot(U, w)
	z = y.reshape(n_states, n_states, actions)
	
	error = np.linalg.norm(y - Y)
	pSpace(z, s='RBF - J={}'.format(J),filename='RBF-Js/RBF-J-{}'.format(J))
	
	return z, error

def SARSA(J=20, epsilon=0.05, QLearn=False, display=True):
	from sklearn.cluster import KMeans

	obs = env.reset()
	
	N = n_states*n_states
	states = np.dstack((pos_grid, vel_grid)).reshape(N, 2)

	sig = np.std(states)
	kmeans = KMeans(n_clusters=J, random_state=0).fit(states)
	
	c = kmeans.cluster_centers_
	w = np.random.randn(J, env.action_space.n).astype(np.longfloat)
	w_hist = np.zeros((episodes, J, env.action_space.n)).astype(np.longfloat)
	total_steps = 0
	
	def QFunc(state, w):
		#perform rbf
		U = np.exp(-np.linalg.norm(state - c, axis=1)/sig)**2
		q = np.dot(U, w)
		return q
		
	def choose_action(state):
		if np.random.uniform(low=0, high=1) < epsilon:
			a2 = np.random.choice(env.action_space.n)
		else:
			a2 = np.argmax(QFunc(state, w))	
		return a2
		
	for episode in range(episodes):
		obs = env.reset()

		total_reward = 0
		alpha = max(min_lr, initial_lr*(gamma**(episode//100)))
		steps = 0

		s1 = obs
		a1 = choose_action(s1)
		while True:
			#env.render()
						
			obs, reward, terminate,_ = env.step(a1)
			total_reward += abs(obs[0]+0.5)
			s2 = obs
			
			Qs1a1 = QFunc(s1, w)[a1]
			I = np.zeros((3, 1))
			I[:a1] = 1
			DQs1a1 = (np.exp(-np.linalg.norm(s1 - c, axis=1)/sig)**2).reshape(-1, 1) @ I.T

			if terminate:
				w += alpha*(reward + Qs1a1) * DQs1a1 
				
				print("Episode:", episode, "Steps:",steps)
				break
			else:
				if QLearn:
					a2 = np.argmax(QFunc(state, w))
				else: 
					a2 = choose_action(s2)

				Qs2a2 = QFunc(s2, w)[a2]
				
				w += alpha*(reward + gamma*Qs2a2 - Qs1a1) * DQs1a1

				s1 = s2
				a1 = a2
				steps += 1
				if(steps%10000 == 0):
					print(steps, 'steps')
				if(np.any(np.isinf(w))):
					print('inf produced')
					exit()
		w_hist[episode] = w
	if display:
		SARSA_anim(w_hist, c, sig, 'QLEARN_RBF' if QLearn else 'SARSA_RBF')
	return w, w_hist
	
def testApprox(q_rbf, lim):
	obs = env.reset()
	steps = 0
	
	while steps<lim:
		pos, vel = discretization(env, obs)
		
		s = np.dstack((pos, vel)).reshape(1, 2)
		
		if np.random.uniform(low=0, high=1) < epsilon:
			a = np.random.choice(env.action_space.n)
		else:
			a = np.argmax(q_rbf[pos][vel])	
		
		obs, reward, terminate,_ = env.step(a)

		steps += 1
		if terminate:
			break
	print("Steps:",steps)
	return steps		

RBFPart = False
SARSAPart = True
	
if(RBFPart):
	#q_table, q_hist = tabQuantize(300, True)
	q_table, q_hist = tabQuantize(1000, False)

	#pSpace(q_table, s='1000 Episode',filename='1000 Episode')
	Js = [10 ,20, 40, 80, 100, 120, 140, 180, 200, 220, 240, 280, 300, 320, 340, 380, 400, 420, 440, 480, 500, 520, 540, 580, 600, 620, 640, 680, 700]
	Es = []
	Ss = []
	for J in Js:
		z, e = RBF_build(J,q_table)
		print(J,':',e)
		Es.append(e)
		Ss.append(testApprox(z, 5000))
		
	fig=plt.figure(figsize=(6,6))
	ax = plt.axes()
	ax.plot(Js, Es)
	ax.set_ylabel("Error")
	ax.set_xlabel("Number of J basis functions")
	fig.savefig('JErrors')
	plt.close(fig)
	
	fig=plt.figure(figsize=(6,6))
	ax = plt.axes()
	ax.plot(Js, Ss)
	ax.set_ylabel("Steps")
	ax.set_xlabel("Number of J basis functions")
	fig.savefig('JSteps')
	plt.close(fig)
	
if(SARSAPart):
	#greedy alg
	#SARSA(J=20, QLearn=True)
	#SARSA
	episodes = 10
	SARSA(J=10)