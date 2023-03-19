import numpy as np
import matplotlib.pyplot as plt
from solvers import *
from plots import plot_results


class BernoulliBandit:
    def __init__(self, k):
        self.probs = np.random.uniform(size=k)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.k = k

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
        

if __name__ == "__main__":
    np.random.seed(1)
    k = 10
    Nstep = 10000
    bandit_10_arm = BernoulliBandit(k)
    print(f"random generated a {k}-armed Bernoulli Bandit")
    print(f"the best arm is {bandit_10_arm.best_idx} with the prob {bandit_10_arm.best_prob}")
    
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(Nstep)
    print("epsilon-greedy solver cumulative regret:", epsilon_greedy_solver.regret)

    plot_results([epsilon_greedy_solver], ['EpsilonGreedy'])

    # epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    # epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
    # epsilon_greedy_solver_names = [f"epsilon={e}" for e in epsilons]

    # for solver in epsilon_greedy_solver_list:
    #     solver.run(Nstep)
    # plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

    decaying_epsilon_greedy_solver = DecayEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(Nstep)
    print('decaying epsilon greedy solver cumulative regret:',
          decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

    ucb_solver = UCB(bandit_10_arm, coef=1)
    ucb_solver.run(Nstep)
    print('ucb solver cumulative regret:', ucb_solver.regret)
    plot_results([ucb_solver], ["UCB"])

    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(Nstep)
    print('thompson sampling solver cumulative regret:', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ['ThompsonSampling'])