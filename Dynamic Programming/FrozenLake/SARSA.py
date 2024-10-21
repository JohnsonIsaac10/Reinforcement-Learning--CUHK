import gym
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random

import rl_utils

env = gym.make('FrozenLake-v1', render_mode="human")
nActions = 4
nStates = env.nrow * env.ncol
Q_table = np.zeros((nStates, nActions))
alpha = 0.3
gamma = 0.9
rewards = []
episodes = 500
epsilon = 0.3


def expect_epsilon(e):
    return min(epsilon, 1.0 - math.log10((e + 1) / 25.))


def expect_alpha(e):
    return min(0.1, 1.0 - math.log10((e + 1) / 125.))


def get_action(state, e):
    if np.random.random() < max(0.001, expect_epsilon(e)):
        return env.action_space.sample()
    a = Q_table
    indices = [i for i, x in enumerate(Q_table[state]) if x == max(Q_table[state])]
    return random.choice(indices)  # choose the maximal Q(s,a)


def take_action(state,e):
    if np.random.random() < expect_epsilon(e):
        action = np.random.randint(nActions)
    else:
        indices = [i for i, x in enumerate(Q_table[state]) if x == max(Q_table[state])]
        action = random.choice(indices)
        # action = np.argmax(Q_table[state])
    return action


def update_SARSA(next_state, reward, action, state, next_action, e):
    Q_next = Q_table[next_state][next_action]
    # if next_state == 2:
    #     print('here')
    Q_table[state][action] += alpha * (
            reward + gamma * (Q_next) - Q_table[state][action])

def print_agent(env, Q_table):
    actions = ["◀", "▼", "▶", "▲"]
    cnt = 1
    for s in range(nStates):
        if s in env.holes:
            print("****", end="   ")
        elif s in env.termination:
            print("Goal", end="   ")
        else:
            # acts = policy[s]
            pi_str = ""
            if s in env.gift_state:
                tmp_idx = list(env.gift_state).index(s)
                pi_str = pi_str + "R"+str(tmp_idx)
            for k in range(len(actions)):
                indices = [i for i, x in enumerate(Q_table[s]) if x == max(Q_table[s])]
                action = random.choice(indices)
                pi_str += actions[k] if k == action else "-"
            print(pi_str, end="   ")
        if cnt % 4 == 0:
            print()
        cnt += 1
    print()


max_reward = 0
teminated = 0
for episode in range(episodes):
    print("----------{}--------".format(episode))
    state = env.reset()
    t = 0
    if type(state) != int:
        state = state[0]
    done = False
    reward_in_ep = 0
    while not done:
        action = take_action(state, episode)
        next_state, reward, done, _, info = env.step(action)
        next_action = take_action(next_state, episode)
        update_SARSA(next_state, reward, action, state, next_action, episode)
        state = next_state
        action = next_action
        t += 1
        reward_in_ep += reward
        if done and reward == 50:
            teminated += 1
        if done and reward_in_ep == 101:
            max_reward += 1
        # if next_state == 2 or next_state == 9:
        #     print('reward: ', reward)

    rewards.append(reward_in_ep)
    print(Q_table)
    print_agent(env, Q_table)
    print('times of max reward: ', max_reward)
    print('times of termination: ', teminated)


print('------------finish-----------------')
print('times of max reward: ', max_reward)
print('times of termination: ', teminated)
np.save('./data/Q_table_SARSA', Q_table)

mv_return = rl_utils.moving_average(rewards, 9)
plt.plot(mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Average returns on {}'.format('FrozenLake-v1'))
plt.savefig('./figures/SARSA_FrozenLake3.png')
plt.show()

