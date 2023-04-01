import numpy as np


class Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha,  gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state][i] == Q_max:
                a[i] = 1
        return a
    
    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1][a1] - self.Q_table[s0][a0]
        self.Q_table[s0][a0] += self.alpha * td_error


class nstep_Sarsa(Sarsa):
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        super().__init__(ncol, nrow, epsilon, alpha, gamma, n_action)
        self.n = n
        self.state_list = []
        self.action_list = []
        self.reward_list = []

    def update(self, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)

        if len(self.state_list) == self.n:
            G = self.Q_table[s1][a1]
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i]
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s][a] += self.alpha * (G - self.Q_table[s][a])
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q_table[s][a] += self.alpha * (G - self.Q_table[s][a])
        if done:
            self.state_list = []
            self.action_list = []
            self.reward_list = []


class QLearning:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self, state):
        q_max = np.max(self.Q_table[state])
        a = [0] * self.n_action
        for i in range(self.n_action):
            if self.Q_table[state][i] == q_max:
                a[i] = 1
        return a
    
    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * np.max(self.Q_table[s1]) - self.Q_table[s0][a0]
        self.Q_table[s0][a0] += self.alpha * td_error
