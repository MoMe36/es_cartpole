import torch 
import torch.nn as nn
import torch.nn.functional as F  
import numpy as np 
import gym 


import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d as gf1d 
import pandas as pd 
from tqdm import tqdm 

from tensorboardX import SummaryWriter
import utils 

class CartPole(gym.Wrapper): 

    def __init__(self): 
        super().__init__(gym.make('CartPole-v0'))

    def process(self, x): 
        return torch.tensor(x).float().reshape(1,-1)
    def reset(self): 
        return self.process(super().reset())
    def step(self, a): 

        ns,r,done, _ = super().step(a)
        return self.process(ns), r, done


class Net(nn.Module): 

    def __init__(self): 

        super().__init__()

        self.l1 = nn.Linear(4, 32)
        self.l2 = nn.Linear(32, 2)

    def forward(self, x): 

        x = F.relu(self.l1(x))
        out = F.softmax(self.l2(x), dim = 1)
        return torch.argmax(out, dim = 1)


def eval_indiv(env, net): 

    s = env.reset()

    reward = 0. 
    done = False
    while not done: 

        with torch.no_grad(): 
            ac = net(s).item()

        s, r, done = env.step(ac)
        reward += r
    return reward 


def sample_noise(net): 

    pos = []
    neg = []

    for p in net.parameters(): 

        noise_t = torch.tensor(np.random.normal(size = p.data.size())).float()
        
        pos.append(noise_t)
        neg.append(-noise_t)

    return pos, neg 

def eval_with_noise(env, net, noise): 

    old_params = net.state_dict()
    for p, p_n in zip(net.parameters(), noise): 
        p.data += noise_std*p_n

    fitness = eval_indiv(env, net)
    net.load_state_dict(old_params)
    return fitness


def train_step(net, batch_noise, batch_reward): 

    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    std_r = np.std(norm_reward)

    if std_r > 1e-6:
        norm_reward /=  std_r

    weighted_noise = None

    for noise, reward in zip(batch_noise, norm_reward): 
        if weighted_noise is None: 
            weighted_noise = [reward*p_n for p_n in noise]
        else: 
            for w_n, p_n in zip(weighted_noise, noise): 
                w_n += reward*p_n

    m_updates = []
    for p, p_update in zip(net.parameters(), weighted_noise): 
        update = p_update/(len(batch_noise)*noise_std)
        p.data += update*lr
        m_updates.append(torch.norm(update).item())
    return m_updates, np.mean(batch_reward), np.std(batch_reward)


writer = utils.make_writer()

net = Net()


max_batch_ep = 50
max_batch_steps = 200
noise_std = 0.01
lr = 0.001
env = CartPole()

p_bar = tqdm(total = max_batch_steps)

recap = {'mean_r': [], 'std_r': [], 'l2_updates': []}

for step in range(max_batch_steps): 

    batch_noise = []
    batch_reward = []
    batch_step = 0

    for i in range(max_batch_ep): 
        noise, neg_noise = sample_noise(net)
        batch_noise.append(noise)
        batch_noise.append(neg_noise)

        p_r = eval_with_noise(env, net,noise)
        n_r = eval_with_noise(env, net, neg_noise)

        batch_reward.extend([p_r, n_r])

    l2_update, mean_r, std_r = train_step(net, batch_noise, batch_reward)
    p_bar.update(1)
    last_r = eval_indiv(env, net)
    p_bar.set_description('Reward: {}'.format(last_r))
    writer.add_scalar('L2 update', np.mean(l2_update), step)
    writer.add_scalar('Mean reward', mean_r, step)
    writer.add_scalar('Reward std', std_r, step)

    recap['mean_r'].append(mean_r)
    recap['l2_updates'].append(np.mean(l2_update))
    recap['std_r'].append(std_r)


p = plt.plot(recap['mean_r'], alpha = 0.3)
plt.plot(gf1d(recap['mean_r'], sigma = 5), color = p[0].get_color())
plt.title('Mean reward')
plt.savefig('mean.png')
plt.cla()

p = plt.plot(recap['l2_updates'], alpha = 0.3)
plt.plot(gf1d(recap['l2_updates'], sigma = 5), color = p[0].get_color())
plt.title('L2 update')
plt.savefig('l2.png')
plt.cla()
p = plt.plot(recap['std_r'], alpha = 0.3)
plt.plot(gf1d(recap['std_r'], sigma = 5), color = p[0].get_color())
plt.title('Std reward')
plt.savefig('std.png')
plt.cla()
p_bar.close() 