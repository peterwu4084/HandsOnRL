import numpy as np


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.k)
        self.regret = 0
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
    
    def run_one_step(self):
        raise NotImplementedError
    
    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1):
        super().__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.k)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.k)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
    

class DecayEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super().__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.k)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.k)
        else:
            k = np.argmax(self.estimates)
        
        r = self.bandit.step(k)
        self.estimates[k] += 1 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
    

class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super().__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.k)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1

        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * self.counts + 2))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1 / (1 + self.counts[k]) * (r - self.estimates[k])
        return k
    

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super().__init__(bandit)
        self._a = np.ones(self.bandit.k)
        self._b = np.ones(self.bandit.k)

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += 1
        self._b[k] += (1 - r)
        return k