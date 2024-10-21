import gym
import numpy as np
import operator
from IPython.display import clear_output
from time import sleep
import random
import itertools
import tqdm
import math
import matplotlib.pyplot as plt

import rl_utils

env = gym.make('FrozenLake-v1', render_mode="human")
tqdm.monitor_interval = 0
random.seed(8983)
inital_epsilon = 0.1
nActions = 4
nStates = env.nrow * env.ncol
rewards_list = []

def expect_epsilon(e):
    return min(inital_epsilon, 1.0 - math.log10((e + 1) / 25))


def create_random_policy(env):
    policy = {}
    for key in range(0, env.observation_space.n):
        p = {}
        for action in range(0, env.action_space.n):
            p[action] = 1 / env.action_space.n
            policy[key] = p
    return policy


def create_state_action_dictionary(env, policy):
    Q = {}
    for key in policy.keys():
        Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q


def play_game(env, policy, display=True):
    """

    :param env:
    :param policy:
    :param display:
    :return: an episode including timesteps, in which each timestep contains state, action, reward
    """
    env.reset()
    episode = []
    finished = False

    while not finished:
        s = env.env.s

        if display:
            clear_output(True)
            env.render()
            sleep(1)

        timestep = []
        timestep.append(s)
        n = random.uniform(0, sum(policy[s].values()))
        top_range = 0
        action = 0
        for prob in policy[s].items():
            top_range += prob[1]
            if n < top_range:
                action = prob[0]
                break
        # print(finished)

        state, reward, finished, _, info = env.step(action)
        # print(state, reward, finished)
        # if state == 2 or state == 9 or state == 27:
        #     print("next state: {}, gift: {}".format(state, reward))
        timestep.append(action)
        timestep.append(reward)

        episode.append(timestep)

    if display:
        clear_output(True)
        env.render()
        sleep(1)
    return episode


def test_policy(policy, env, episodes):
    wins = 0
    total_reward = 0

    print("---------testing---------")
    termination = 0
    for i in range(episodes):
        print("testing episode------------{}-------------".format(i))

        test = play_game(env, policy, display=False)
        w = test[-1][-1]
        test_array = np.array(test)
        # total_reward = total_reward + np.sum(test_array, axis=0)[-1]
        G = 0           # episode_reward

        for i in reversed(range(0, len(test))):
            s_t, a_t, r_t = test[i]
            if r_t == 50:
                termination += 1
            G += r_t
        # print(i)

        # maximum
        if G == 101:
            wins += 1
        total_reward += G
    average_reward = total_reward / episodes
    return wins, average_reward, termination


def evaluate_policy_check(env, episode, policy, test_policy_freq):
    print("here")
    print(episode)
    if episode % test_policy_freq == 0:
        print("Test policy for episode {} wins % = {}"
              .format(episode, test_policy(policy, env)))


def monte_carlo_e_soft(env, episodes=5, policy=None, epsilon=0.01, test_policy_freq=1000):
    """

    :param env:
    :param episodes:
    :param policy: dictionary. Keys are states, items are (action, prob) pairs for every action.
    :param epsilon:
    :param test_policy_freq:
    :return:
    """
    if not policy:
        policy = create_random_policy(env)
    Q = create_state_action_dictionary(env, policy)
    returns = {}

    count_gift_state = 0
    count_gift_reward = 0

    for e in range(episodes):
        print("episode------------{}-------------".format(e))
        G = 0
        env.reset()
        episode = play_game(env=env, policy=policy, display=False)

        # evaluate_policy_check(env, e, policy, test_policy_freq)
        for i in reversed(range(0, len(episode))):
            s_t, a_t, r_t = episode[i]
            state_action = (s_t, a_t)
            G += r_t
            # print(i)
            a = episode[0:i]
            # print("outside: ", i)
            # if r_t == 1:
            #     count_gift_reward += 1
            if not s_t in [x[0] for x in episode[0:i]]:
                if s_t == 2:
                    count_gift_state += 1

            if not state_action in [(x[0], x[1]) for x in episode[0:i]]:
                # print(i)

                # if the state-action pair is not visited, append it.
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]

                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action])
                # print('s_t: ', s_t)

                # test = Q[s_t]

                # Q: keys are the states, item under each key is (action, reward)

                # Q_list is the reward list under state s_t
                Q_list = list(map(lambda x: x[1], Q[s_t].items()))
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                max_Q = random.choice(indices)  # choose the maximal Q(s,a)

                A_star = max_Q
                for a in policy[s_t].items():
                    if a[0] == A_star:
                        policy[s_t][a[0]] = 1 - expect_epsilon(e) + (epsilon / abs(sum(policy[s_t].values())))
                    else:
                        policy[s_t][a[0]] = (expect_epsilon(e) / abs(sum(policy[s_t].values())))
                # for a in policy[s_t].items():
                #     if a[0] == A_star:
                #         policy[s_t][a[0]] = 1 - (1/(e+1)) + ((1/(e+1)) / abs(sum(policy[s_t].values())))
                #     else:
                #         policy[s_t][a[0]] = ((1/(e+1)) / abs(sum(policy[s_t].values())))
        print("G: ", G)
        count_gift_reward = G + count_gift_reward
        rewards_list.append(G)
        print_agent(env, policy)
    return policy


def print_agent(env, policy):
    actions = ["◀", "▼", "▶", "▲"]
    cnt = 1
    for s in range(nStates):
        n = random.uniform(0, sum(policy[s].values()))
        top_range = 0
        action = 0
        for prob in policy[s].items():
            top_range += prob[1]
            if n < top_range:
                action = prob[0]
                break

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
                pi_str += actions[k] if k == action else "-"

            print(pi_str, end="   ")
        if cnt % 4 == 0:
            print()
        cnt += 1
    print()

# env = gym.make('FrozenLake-v0')
policy = monte_carlo_e_soft(env, episodes=300)

print(policy)
print_agent(env, policy)
mv_return = rl_utils.moving_average(rewards_list, 9)
plt.plot(mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Average returns on {}'.format('FrozenLake-v1'))
plt.savefig('./figures/MonteCalo_FrozenLake3.png')
plt.show()

print("Testing......")
wins, average_reward, termination = test_policy(policy, env, episodes=100)
print("Times of maximal reward: ", wins)
print("Times of termination: ", termination)
print("Average reward: ", average_reward)
print('----------end-----------')
