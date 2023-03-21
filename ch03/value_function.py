import numpy as np


def compute(P, rewards, gamma, states_num):
    rewards = np.array(rewards).reshape((-1, 1))
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value


if __name__ == "__main__":
    P = [
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
        [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ]

    P = np.array(P)
    rewards = [-1, -2, -2, 10, 1, 0]
    gamma = 0.5

    V = compute(P, rewards, gamma, 6)
    print("each value is:")
    print(V)