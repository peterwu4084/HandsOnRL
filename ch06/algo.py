import numpy as np
import random


class DynaQ:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_planning, n_action=4):
        self.Q_table = np.zeros([ncol * nrow, n_action])
        self.n_action = n_action
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.n_planning = n_planning
        self.model = dict()

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * max(self.Q_table[s1]) - self.Q_table[s0][a0]
        self.Q_table[s0][a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1
        for _ in  range(self.n_planning):
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)

        