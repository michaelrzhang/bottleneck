
import pickle
import matplotlib.pyplot as plt
import numpy as np
def plot_mis(data, reward_data = None):
	num_mi, hidden_units = data.shape
	if reward_data:
		f, axes = plt.subplots(2, 1, sharex=True, figsize=(10,10))
		axes[0].set_title("Mutual information between hidden units and input")
		# TODO: fix
		# for i in range(hidden_units // 2):
		# 	axes[0].plot(range(num_mi), data[:, i], label = str(i), c = "red")
		# for i in range(hidden_units // 2 + 1, hidden_units):
		# 	axes[0].plot(range(num_mi), data[:, i], label = str(i), c = "blue")

		first_x = np.mean(data[:, :3], axis =1)
		second_x = np.mean(data[:, 3:6], axis =1)
		entropies = np.mean(data[:, 6:9], axis = 1)
		all_x = np.mean(data[:, 9:12], axis = 1)
		axes[0].plot(range(num_mi), first_x, c = "red", label="irrelevant I(X1;T)")
		axes[0].plot(range(num_mi), second_x, c = "blue", label = "direction I(X2;T) = I(Y;T)")
		axes[0].plot(range(num_mi), entropies, c = "green", label= "entropy H(T)")
		axes[0].plot(range(num_mi), entropies, c = "orange", label= "I(X;T)")
		axes[0].set_ylim((0, 3))
		axes[0].legend()



		axes[1].set_title("Cumulative reward")
		axes[1].set_xlabel("Hundreds of episodes")
		reward_data = np.array(reward_data)
		axes[1].plot(reward_data[:,0] // 100., reward_data[:,1])
		f.savefig("data/plot")
	else:
		for i in range(hidden_units):
			plt.plot(range(num_mi), data[:, i], label = str(i))
	plt.show()

data_folder = "data/directionlatest_mi.p"
reward_data_folder = "data/directionreward_sums.p"

# data_folder = "data/simplelatest_mi.p"
# reward_data_folder = "data/simplereward_sums.p"

data = pickle.load(open(data_folder, "rb"))
reward_data = pickle.load(open(reward_data_folder, "rb"))
# print(reward_data)
# import ipdb as pdb; pdb.set_trace()
print("hi")
plot_mis(data, reward_data)