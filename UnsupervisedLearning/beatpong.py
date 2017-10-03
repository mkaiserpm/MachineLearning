#Python tutorial unsupervised learning
#Mario Kaiser 2017
# taken from sirlogy live session "How to beat pong using policy gradients"
#Video: https://www.youtube.com/watch?v=PDbXPBwOavc
#Github Code: https://github.com/llSourcell/Policy_Gradients_to_beat_Pong/blob/master/demo.py

import numpy as np 
import cPickle as pickle 
import gym #openai environment

#hyperparameters
# use grid search or parameters published in papers

H = 200 # size of hiddenlayers
batch_size = 10
learning_rate = 1e-4
gamma = 0.99 #discount factor reward function (promote short term)
decay_rate = 0.99 #used for differentiation
resume = False # 

#init model
D = 80 * 80 #80 by 80 pixel frame from pong

if resume:
	model = pickle.load(open('save.p', 'rb'))
else:
	model = {}
	#xavier initialisation , takes hidden nodes into account
	model['W1'] = np.random.randn(H,D) / np.sqrt(D)
	model['W2'] = np.random.randn(H) / np.sqrt(H)
grad_buffer = {k : np.zeros_like(v) for k,v in model.iteritems()}
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems()}

#activation function
def sigmoid(x):
	return 1. / (1.+ np.exp(-x))

def prepro(I): #input preparation pixel image
	I = I[35:195] #crop (magic number that have been investigated before)
	I = I[::2,::2,0]
	I = [I==144] = 0 #erase background
	I = [I==109] = 0
	I[I !=0 ] = 1 #paddles and balls set to 1
	return I.astype(np.float).ravel() #flattens matrix

#short term reward 
def discount_reward(r): 
	discounted_r = np.zeros_like(r) #reinforcement zero to all discount reward
	#rewards are expentionally decreased in value for long term
	running_add = 0 #sum of the rewards
	for t in reversed(xrange(0, r.size)):
		if r[t] != 0: running_add = 0
		#increment sum
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

def policy_forward(x):
	#forward propagation
	h = np.dot(model['W1'],x)
	h[ h < 0 ] = 0 #Relu
	logp = np.dot(model['W2'], h)
	p = sigmoid(logp)
	return p,h #return probality of taking action 2 and hidden state

def policy_backward(eph,epdlogp):
	#eph is array of intermediate states
	dW2 = np.dot(eph.T, epdlogp).ravel()
	dh = np.outer(epdlogp,model['W2'])
	dh[eph <= 0.] = 0. #Relu
	dW1 = np.dot(dh.T,epx)
	#return both derivatise to update weights
	return {'W1':dW1,'W2':dW2}

#implementation
env = gym.make('Pong-v0')
observation = env.reset()
prev_x = None

#calculate difference between frame
xs, hs, dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

#begin training
while True:

	# preprocess the observation, set input to network to be difference image
    #Since we want our policy network to detect motion
    #difference image = subtraction of current and last frame
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    #so x is our image difference, feed it in!

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    #this is the stochastic part 
    #since not apart of the model, model is easily differentiable
    #if it was apart of the model, we'd have to use a reparametrization trick (a la variational autoencoders. so badass)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    y = 1 if action == 2 else 0 # a "fake label"
    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    env.render()
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    if done:

        # an episode finished
	    episode_number += 1

	    # stack together all inputs, hidden states, action gradients, and rewards for this episode
	    #each episode is a few dozen games
	    epx = np.vstack(xs) #obsveration
	    eph = np.vstack(hs) #hidden
	    epdlogp = np.vstack(dlogps) #gradient
	    epr = np.vstack(drs) #reward
	    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

	    #the strength with which we encourage a sampled action is the weighted sum of all rewards afterwards, but later rewards are exponentially less important
	    # compute the discounted reward backwards through time
	    discounted_epr = discount_rewards(epr)
	    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
	    discounted_epr -= np.mean(discounted_epr)
	    discounted_epr /= np.std(discounted_epr)

	    #advatnage - quantity which describes how good the action is compared to the average of all the action.
	    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
	    grad = policy_backward(eph, epdlogp)
	    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

	    # perform rmsprop parameter update every batch_size episodes
	    #http://68.media.tumblr.com/2d50e380d8e943afdfd66554d70a84a1/tumblr_inline_o4gfjnL2xK1toi3ym_500.png
	    if episode_number % batch_size == 0:
	      for k,v in model.iteritems():
	        g = grad_buffer[k] # gradient
	        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
	        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
	        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

	    # boring book-keeping
	    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
	    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
	    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
	    reward_sum = 0
	    observation = env.reset() # reset env
	    prev_x = None

	if reward != 0:
	  	# Pong has either +1 or -1 reward exactly when game ends.
	    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')

