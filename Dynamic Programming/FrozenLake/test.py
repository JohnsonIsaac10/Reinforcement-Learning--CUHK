import numpy as np
import gym
import matplotlib.pyplot as plt


class BernoulliBandit:
    def __init__(self, K):
        self.probs = np.random.uniform(low=0, high=1, size=K)  # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
        # 概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.K = K

    def step(self, k):
        # 当玩家选择了k号拉杆后,p概率返回1（获奖），1-p概率返回0（未获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


class MAB:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 记录每一步的动作
        self.regrets = []  # 记录每一步的累积懊悔

    def update_regret(self, k):
        # 计算累积懊悔, k为选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆
        # 在新的类重写
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class DecayingEpsilonGreedy(MAB):
    """ decaying-epsilon greedy algorithm"""
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)  # 初始化拉动所有拉杆的期望奖励估值
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


def plot_results(solvers, solver_names):

    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Accumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)

    plt.legend()
    plt.savefig('./multi_arm_bandit.png')
    plt.show()


np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)

np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('accumulative regrets of decaying-epsilon algorithm：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])
# gym.envs.registry.all()

