# from os import system, name
# import time
# import gym
# import numpy as np
#
# env = gym.make('FrozenLake-v1')
# env.reset()
#
#
# def clear():
#     if name == 'nt':
#         _ = system('cls')
#     else:
#         _ = system('clear')
#
#
# def act(V, env, gamma, policy, state, v):
#     for action, action_prob in enumerate(policy[state]):
#         for state_prob, next_state, reward, end in env.P[state][action]:
#             v += action_prob * state_prob * (reward + gamma * V[next_state])
#             V[state] = v
#
#
# def eval_policy(policy, env, gamma=1.0, theta=1e-9, terms=1e9):
#     V = np.zeros(env.nS)
#     delta = 0
#     for i in range(int(terms)):
#         for state in range(env.nS):
#             act(V, env, gamma, policy, state, v=0.0)
#         clear()
#         print(V)
#         time.sleep(1)
#         v = np.sum(V)
#         if v - delta < theta:
#             return V
#         else:
#             delta = v
#     return V
#
#
# policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA
# V = eval_policy(policy, env.env)
#
# print(policy, V)

import gym
import numpy as np
import time


env = gym.make("FrozenLake-v1", render_mode="human")
env.action_space.seed(42)

nActions = 4

nStates = env.nrow * env.ncol

observation, info = env.reset(seed=42)

def act(V, env, gamma, policy, state, v):
    for action, action_prob in enumerate(policy[state]):
        for state_prob, next_state, reward, end in env.P[state][action]:
            ll = env.P
            v += action_prob * state_prob * (reward + gamma * V[next_state])
            V[state] = v


def eval_policy(policy, env, gamma=1.0, theta=1e-9, terms=1e9):
    V = np.zeros(nStates)
    delta = 0
    for i in range(int(terms)):
        for state in range(nStates):
            act(V, env, gamma, policy, state, v=0.0)
        # print(V)
        time.sleep(1)
        v = np.sum(V)
        if v - delta < theta:
            return V
        else:
            delta = v
    return V

P = {s: {a: [] for a in range(nActions)} for s in range(nStates)}

a = np.eye(nActions)[1]

policy = np.ones([nStates, nActions]) / nActions
V = eval_policy(policy, env)



print(policy)
# for _ in range(1000):
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
#
#     if terminated or truncated:
#         print("here is timestep ", _)
#         print(observation)
#         observation, info = env.reset()

env.close()

