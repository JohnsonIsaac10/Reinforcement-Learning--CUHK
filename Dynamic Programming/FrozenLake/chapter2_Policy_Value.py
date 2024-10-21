import random
from os import system, name
import time
import gym
import numpy as np
import math
from math import pow
import copy
import itertools as it

env = gym.make('FrozenLake-v1', render_mode="human")
env.reset()

nActions = 4

nStates = env.nrow * env.ncol



def act(V, env, gamma, policy, state, v):
    for action, action_prob in enumerate(policy[state]):

        for state_prob, next_state, reward, end in env.P[state][action]:
            v += action_prob * state_prob * (reward + gamma * V[next_state])
            V[state] = v


def evaluate(V, action_values, env, gamma, state):
    for action in range(nActions):
        for prob, next_state, reward, terminated in env.P[state][action]:
            action_values[action] += prob * (reward + gamma * V[next_state])
    return action_values


def Q_function(env, state, V, gamma):
    action_values = np.zeros(nActions)
    return evaluate(V, action_values, env, gamma, state)


def improve_policy(env, gamma=0.9, terms=200):
    policy = np.ones([nStates, nActions]) / nActions
    evals = 1
    V = None
    for i in range(int(terms)):
        # stable = True
        V = eval_policy(policy, env, gamma=gamma)
        print("policy iteration: {}".format(i))
        old_policy = copy.deepcopy(policy)
        for state in range(nStates):
            # indices = [i for i, x in enumerate(policy[state]) if x == max(policy[state])]
            # current_action = random.choice(indices)
            # current_action = np.argmax(policy[state])
            action_value = Q_function(env, state, V, gamma)
            maxq = max(action_value)
            cntq = np.sum(action_value==maxq)
            policy[state] = [1/cntq if q == maxq else 0 for q in action_value]
            # indices = [i for i, x in enumerate(action_value) if x == max(action_value)]
            # best_action = random.choice(indices)
            # best_action = np.argmax(action_value)

            # if current_action != best_action:
            #     # stable = False
            #     policy[state] = np.eye(nActions)[best_action]
            #
            # evals += 1
            # if stable:
            #     return policy, V
        if old_policy.all() == policy.all():
            return policy, V
    return policy, V



def eval_policy(policy, env, gamma=0.9, theta=1e-5, terms=100000):
    V = np.zeros(nStates)
    delta = 0
    for i in range(int(terms)):
        for state in range(nStates):
            act(V, env, gamma, policy, state, v=0.0)
        # print(V)
        # time.sleep(1)
        v = np.sum(V)
        if v - delta < theta:
            return V
        else:
            delta = v
    return V


def value_iteration(env, gamma=0.9, theta=1e-9, terms=100000):
    V = np.zeros(nStates)
    # print('value iteration')
    for i in range(int(terms)):
        delta = 0
        print("value iteration: {}".format(i))
        for state in range(nStates):
            action_value = Q_function(env, state, V, gamma)
            best_action_value = np.max(action_value)
            delta = max(delta, np.abs(V[state] - best_action_value))
            V[state] = best_action_value
        if delta < theta: break
    policy = np.zeros([nStates, nActions])
    for state in range(nStates):
        action_value = Q_function(env, state, V, gamma)
        # best_action = np.argmax(action_value)
        # policy[state, best_action] = 1.0
        maxq = max(action_value)
        cntq = np.sum(action_value == maxq)
        policy[state] = [1 / cntq if q == maxq else 0 for q in action_value]
    return policy, V


def play(env, episodes, policy_list):
    wins = 0
    total_reward = 0
    n = len(env.gift_state)
    gift_state = list(env.gift_state)
    termination = 0
    for episode in range(episodes):
        term = False
        state = env.reset()
        reward_in_episode = 0
        policy_idx = list(it.product(range(2), repeat=n))[0]
        state_idx = list(it.product(range(2), repeat=n))[0]

        while not term:
            # 若到达gift位置，则更换策略
            if type(state) != int:
                state = state[0]

            if state in gift_state:
                tmp_idx = gift_state.index(state)
                state_idx = list(state_idx)
                state_idx[tmp_idx] = state
                state_idx = tuple(state_idx)
                policy_idx = state_idx

            policy = policy_list[policy_idx]

            a = policy[state]

            indices = [i for i, x in enumerate(a) if x == max(a)]
            action = random.choice(indices)
            # action = np.argmax(a)
            next_state, reward, term, _, info = env.step(action)
            total_reward += reward
            state = next_state
            reward_in_episode += reward
            # if state == 2 or state == 9 or state == 27:
            #     print("next state: {}, gift: {}".format(state, reward))
            if term and reward_in_episode == 101:
                wins += 1
            if term and reward == 50:
                termination += 1
    average_reward = total_reward / episodes
    return wins, total_reward, average_reward, termination


def update_matrix(env):
    gift_state = env.gift_state

    policy_plc_list = {}    # policies generated by policy iteration
    value_plc_list = {}     # values generated by policy iteration

    policy_val_list = {}    # policies generated by value iteration
    value_val_list = {}     # values generated by value iteration

    n = len(gift_state)

    a = 2**n

    state_idx = list(it.product(range(2),repeat=n))
    test = []

    for s_idx in state_idx:
        c = tuple(l*r for l, r in zip(s_idx, list(gift_state)))
        for element in c:
            if element != 0:
                env.update_giftstate_neighbour(element, env.gift_reward_done)
        test.append(c)

        policy_plc, value_plc = improve_policy(env)
        policy_val, value_val = value_iteration(env)

        policy_plc_list[c] = policy_plc
        value_plc_list[c] = value_plc

        policy_val_list[c] = policy_val
        value_val_list[c] = value_val

        # recover
        for element in c:
            if element != 0:
                env.update_giftstate_neighbour(element, env.gift_reward)

    # recover
    for gift_s in gift_state:
        env.update_giftstate_neighbour(gift_s, env.gift_reward)

    return policy_plc_list, value_plc_list, policy_val_list, value_val_list


def save_policy(env):

    policy_plc_list, value_plc_list, policy_val_list, value_val_list = update_matrix(env)

    np.save('./save_policy/policy_plc_list', policy_plc_list)
    np.save('./save_policy/value_plc_list', value_plc_list)

    np.save('./save_policy/policy_val_list', policy_val_list)
    np.save('./save_policy/value_val_list', value_val_list)

    # np.save('./save_policy/policy_plc_list(no_gift)', policy_plc_list)
    # np.save('./save_policy/value_plc_list(no_gift)', value_plc_list)
    #
    # np.save('./save_policy/policy_val_list(no_gift)', policy_val_list)
    # np.save('./save_policy/value_val_list(no_gift)', value_val_list)


def load_policy():
    policy_plc_list = np.load('./save_policy/policy_plc_list.npy', allow_pickle=True)
    value_plc_list = np.load('./save_policy/value_plc_list.npy', allow_pickle=True)

    policy_val_list = np.load('./save_policy/policy_val_list.npy', allow_pickle=True)
    value_val_list = np.load('./save_policy/value_val_list.npy', allow_pickle=True)


    # policy_plc_list = np.load('policy_plc_list(no_gift).npy', allow_pickle=True)
    # value_plc_list = np.load('value_plc_list(no_gift).npy', allow_pickle=True)
    #
    # policy_val_list = np.load('policy_val_list(no_gift).npy', allow_pickle=True)
    # value_val_list = np.load('value_val_list(no_gift).npy', allow_pickle=True)

    policy_plc_list = policy_plc_list.item()
    value_plc_list = value_plc_list.item()

    policy_val_list = policy_val_list.item()
    value_val_list = value_val_list.item()

    return policy_plc_list, value_plc_list, policy_val_list, value_val_list


def print_agent(env, policy_list):
    actions = ["◀", "▼", "▶", "▲"]
    cnt = 1
    for key_value in policy_list.items():
        key = key_value[0]
        policy = key_value[1]
        tmp_str = "Reward "

        for i in range(len(key)):
            tmp_str += str(i)+', ' if key[i]>0 else "-, "
        print(tmp_str+"picked")
        for s in range(nStates):
            if s in env.holes:
                print("****", end="   ")
            elif s in env.termination:
                print("Goal", end="   ")
            else:
                acts = policy[s]
                pi_str = ""
                if s in env.gift_state:
                    tmp_idx = list(env.gift_state).index(s)
                    pi_str = pi_str + "R"+str(tmp_idx)
                for k in range(len(actions)):
                    pi_str += actions[k] if acts[k] > 0 else "-"
                print(pi_str, end="   ")
            if cnt % 4 == 0:
                print()
            cnt += 1
        print()

if __name__ == '__main__':

    # save_policy(env)

    # gift2 = 35, gift9 = 16, goal = 50
    policy_plc_list, value_plc_list, policy_val_list, value_val_list = load_policy()
    print("-----------policy iteration-----------")
    # print('policy after policy iteration: ')
    # print(policy_plc_list)
    # print('value function after policy iteration: ')
    # print(value_plc_list)

    print_agent(env, policy_plc_list)
    wins_plc, total_plc, avg_plc, termination_plc = play(env, 100, policy_plc_list)
    print('Times of getting maximal reward: ', wins_plc)
    print('Times of reaching the termination: ', termination_plc)
    print('Average reward: ', avg_plc)

    print('\n')

    print("-----------value iteration------------")
    print_agent(env, policy_plc_list)
    # print('policy after value iteration: ')
    # print(policy_val_list)
    # print('value function after value iteration: ')
    # print(value_val_list)

    wins_val, total_val, avg_val, termination_val = play(env, 100, policy_val_list)
    print('Times of getting maximal reward: ', wins_val)
    print('Times of reaching the termination: ', termination_val)
    print('Average reward: ', avg_val)
    # print(wins_val, avg_val, termination_val)


