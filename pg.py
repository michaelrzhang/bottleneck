""" Trains an agent with (stochastic) Policy Gradients. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
import pdb
from layers import *
from frozen import FrozenSimpleEnv
from entropy_estimators.est_MI import est_MI_JVHW
from entropy_estimators.est_entro import est_entro_JVHW
import matplotlib.pyplot as plt

#TODOs: 
# add bias?
# abstract away backprop math/ make sure backprop is correct
# reduce batch size?
# different slipperiness on report
# hyperparameters

batch_size = 100 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.95 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
max_episodes = 8e4
np.random.seed(2)

# env = gym.make('FrozenLake-v0')
# num_actions = 4 # out dim
remove_bad_episodes = False
bad_episodes_removed = 0.5
num_bins = 10

# env_type = "simple"
env_type = "direction"
if env_type == "simple":
    env = FrozenSimpleEnv(randomness=0.75)
    num_actions = 2 # out dimension
    D = 1 # input layer dimension
    H = 3 # number of hidden layer neurons
elif env_type == "direction":
    env = FrozenSimpleEnv(randomness=0.75, constant_direction = False)
    num_actions = 2
    D = 2
    H = 3
else:
    print("not supported!")

log_freq = 5 # prints out information every log_freq batches
compute_entropy = True

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
    model['B1'] = np.zeros((H, 1)) * 0.1 
    model['W2'] = np.random.randn(num_actions, H) / np.sqrt(H)
    model['B2'] = np.zeros((num_actions, 1)) * 0.1 
    
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        # if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    # pdb.set_trace()
    h = np.dot(model['W1'], x) + model['B1'].flatten()
    h = sigmoid(h) 
    # pdb.set_trace()
    logp = np.dot(model['W2'], h) + model['B2'].flatten()
    p = softmax_forward(logp)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    # pdb.set_trace()

    dW2 = np.dot(eph.T, epdlogp)
    # pdb.set_trace()
    dB2 = np.mean(epdlogp, axis = 0)
    dh = np.dot(epdlogp, model['W2'])
    dh = eph * (1 - eph) * dh 
    
    dB1 = np.mean(dh, axis = 0)
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2.T, 'B1': dB1.reshape(len(dB1),1), 'B2': dB2.reshape(len(dB2),1)}

def compute_mi_x(x, h, bins = 20):
    """
    takes as input x and h
    """
    bins = [i * 1.0 / bins for i in range(bins + 1)]
    # iterate over hidden units
    MIs = []
    entropy = []
    mi_xys = []
    # pdb.set_trace()
    # first i is basically same as Y, second is useless I(x;T) shoudl go down
    uu = x[:, 1] * 5 + x[:, 0]

    for i in range(x.shape[1]):
        for j in range(h.shape[1]):
            y = np.digitize(h[:, j], bins)
            x_var = x[:, i]
            mi = est_MI_JVHW(x_var, y)
            mi_xy = est_MI_JVHW(uu, y)[0]
            entro = est_entro_JVHW(y)[0]
            entropy.append(entro)
            mi_xys.append(mi_xy)

            # entro = est_entro_JVHW(y) # y should be a function of x, hence..
            # assert (entro[0] - mi[0]) < 1e-4
            MIs.append(mi[0])

    # pdb.set_trace()
    MIs.extend(entropy[:3])
    MIs.extend(mi_xys[:3])
    MIs.append(est_entro_JVHW(uu)[0])
    return MIs # last three terms are entropy

observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]

if compute_entropy:
    all_x = []
    all_h = []
    all_MIs = []
all_reward_sums = []

running_reward = None
reward_sum = 0
episode_number = 0
total_trials = 0
successful_trials = 0

while episode_number < max_episodes:
    if render: env.render()

    # forward the policy network and sample an action from the returned probability
    x = observation
    # import ipdb; ipdb.set_trace()
    aprob, h = policy_forward(x)
    h = h.reshape(1, H) # makes shapes align
    aprob = aprob.reshape(num_actions)
    # action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
    action = np.random.choice(np.arange(0, num_actions), p = aprob)
    label = np.zeros(num_actions)
    label[action] = 1 #fake label 

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    if compute_entropy:
        all_h.append(h.flatten())
        all_x.append(x)
    dlogps.append(label - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    if done: # an episode finished
        episode_number += 1
        total_trials += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        # print(epx.shape, eph.shape, epdlogp.shape, epr.shape) # debugging shapes
        # if length 3 episode (3, 1) (3, 10) (3, 4) (3, 1)

        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        
        # option to skips trials where reward is entirely 0
        if remove_bad_episodes:
            if np.std(discounted_epr) == 0 and (episode_number % batch_size) < batch_size * bad_episodes_removed:
                observation = env.reset()
                episode_number -= 1
                continue
        
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        # discounted_epr /= np.std(discounted_epr).

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # gradient update
        if episode_number % batch_size == 0:
            for k,v in model.items():
                # Vanilla Gradient
                g = grad_buffer[k] # gradient
                model[k] += learning_rate * g
                print(episode_number)
                # # RMSprop
                # rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                # model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                # grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            if compute_entropy:
                all_h = np.array(all_h)
                all_x = np.array(all_x)
                print("Computing mutual information with {} samples.. ".format(len(all_x)))
                MIs = compute_mi_x(h=all_h, x=all_x, bins =num_bins)
                all_MIs.append(MIs)
                print(MIs)
                all_h = []
                all_x = []
       
        # log
        if episode_number % (batch_size * log_freq) == 0: 
            pickle.dump(model, open('data/lastest_model.p', 'wb'))
            rate = successful_trials * 1.0 / total_trials
            print("episodes: {} avg_reward: {}".format(episode_number, reward_sum / (batch_size * log_freq)))
            all_reward_sums.append((episode_number, reward_sum / (batch_size * log_freq)))
            
            reward_sum = 0
            total_trials = 0
            

 

        # reward_sum = 0
        observation = env.reset() # reset env
if compute_entropy:
    all_MIs = np.array(all_MIs)
    pickle.dump(all_MIs, open('data/{}latest_mi.p'.format(env_type), "wb"))
    pickle.dump(all_reward_sums, open('data/{}reward_sums.p'.format(env_type), "wb"))
