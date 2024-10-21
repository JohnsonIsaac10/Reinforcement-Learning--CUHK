import math, random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import collections

import matplotlib.pyplot as plt

import gym
import numpy as np

from collections import deque
from tqdm import trange


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


env_id = "CartPole-v0"
env = gym.make(env_id)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

eps_by_episode = lambda episode: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
    -1. * episode / epsilon_decay)

plt.plot([eps_by_episode(i) for i in range(10000)])
plt.show()


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            # state = autograd.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            state = torch.tensor(state, dtype=torch.float32)
            q_value = self.forward(state)
            action = q_value.argmax().item()
        else:
            action = random.randrange(env.action_space.n)

        # if np.random.random() < epsilon:
        #     action = np.random.randint(env.action_space.n)
        # else:
        #     state = torch.tensor([state], dtype=torch.float)
        #     q_value = self.forward(state)
        #     action = q_value.argmax().item()
        return action


model = DQN(env.observation_space.shape[0], env.action_space.n)

optimizer = optim.Adam(model.parameters())

replay_buffer = ReplayBuffer(1000)


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    # state = autograd.Variable(torch.Tensor(state))
    # next_state = autograd.Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    # action = autograd.Variable(torch.LongTensor(action))
    # reward = autograd.Variable(torch.FloatTensor(reward))
    # done = autograd.Variable(torch.FloatTensor(done))

    state = torch.tensor(state, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    action = torch.tensor(action).view(-1,1)
    reward = torch.tensor(reward, dtype=torch.float32).view(-1,1)
    done = torch.tensor(done, dtype=torch.float32).view(-1,1)

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    # loss = (q_value - autograd.Variable(expected_q_value.data)).pow(2).mean()
    loss = torch.mean(F.mse_loss(q_value, expected_q_value))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def plot(episode, rewards, losses):
    # clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('episode %s. reward: %s' % (episode, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


episodes = 10000
batch_size = 64
gamma = 0.99

losses = []
all_rewards = []
episode_reward = 0

tot_reward = 0
tr = trange(episodes + 1, desc='Agent training', leave=True)
for episode in tr:
    state = env.reset()
    # print('------------{}-------------'.format(episode))
    tr.set_description("Agent training (episode{}) Avg Reward {}".format(episode + 1, tot_reward / (episode + 1)))
    tr.refresh()
    if type(state) != int:
        state = state[0]
    # if episode == 0:
    #     state = state[0]
    done = False
    while not done:
        epsilon = eps_by_episode(episode)
        action = model.act(state, epsilon)

        next_state, reward, done, _, info = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)

        tot_reward += reward

        state = next_state
        episode_reward += reward

    # print(reward)
    print(episode_reward)
    if done:
        # state = env.reset()
        all_rewards.append(episode_reward)
        # print('-----------------------')
        episode_reward = 0

    if replay_buffer.size() > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())

    if episode % 2000 == 0:
        # print(all_rewards)
        # print(losses)
        plot(episode, all_rewards, losses)