import numpy as np
import matplotlib.pyplot as plt
from solvers import *


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
    bandit_10_arm = BernoulliBandit(k)
    print(f"random generated a {k} arm Bernoulli Bandit")
    print(f"the best arm is {bandit_10_arm.best_idx} with the prob {bandit_10_arm.best_prob}")
    