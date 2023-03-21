import numpy as np
from value_function import compute


def join(str1, str2):
    return str1 + '-' + str2


def sample(MDP, Pi, timestep_max, number):
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes


def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode)-1, -1, -1):
            s, a, r, s_next = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + 1 / N[s] * (G - V[s])

    
def occupancy(episodes, s, a, timestep_max, gamma):
    rho = 0
    total_times = np.zeros(timestep_max)
    occur_times = np.zeros(timestep_max)
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho = occur_times[i] / total_times[i] + gamma * rho
    return (1 - gamma) * rho


if __name__ == "__main__":
    S = ['s1', 's2', 's3', 's4', 's5', 's6']
    A = ["keep_s1", "to_s1", "to_s2", "to_s3", "to_s4", "to_s5", "prob_to"]
    P = {
        "s1-keep_s1-s1": 1.0, "s1-to_s2-s2": 1.0,
        "s2-to_s1-s1": 1.0, "s2-to_s3-s3": 1.0,
        "s3-to_s4-s4": 1.0, "s3-to_s5-s5": 1.0,
        "s4-to_s5-s5": 1.0, "s4-prob_to-s2": 0.2,
        "s4-prob_to-s3": 0.4, "s4-prob_to-s4": 0.4
    }
    R = {
        "s1-keep_s1": -1, "s1-to_s2": 0,
        "s2-to_s1": -1, "s2-to_s3": -2,
        "s3-to_s4": -2, "s3-to_s5": 0,
        "s4-to_s5": 10, "s4-prob_to": 1
    }

    gamma = 0.5
    MDP = (S, A, P, R, gamma)

    # policy 1, random policy
    Pi_1 = {
        "s1-keep_s1": 0.5, "s1-to_s2": 0.5,
        "s2-to_s1": 0.5, "s2-to_s3": 0.5,
        "s3-to_s4": 0.5, "s3-to_s5": 0.5,
        "s4-to_s5": 0.5, "s4-prob_to": 0.5
    }

    # policy 2
    Pi_2 = {
        "s1-keep_s1": 0.6, "s1-to_s2": 0.4,
        "s2-to_s1": 0.3, "s2-to_s3": 0.7,
        "s3-to_s4": 0.5, "s3-to_s5": 0.5,
        "s4-to_s5": 0.1, "s4-prob_to": 0.9
    }

    P_from_mdp_to_mrp = [
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.5],
        [0.0, 0.1, 0.2, 0.2, 0.5],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ]
    P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
    R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

    V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
    print("each value is")
    print(V)

    episodes = sample(MDP, Pi_1, 20, 1000)
    print('1st episode\n', episodes[0])
    print('2nd episode\n', episodes[1])
    print('3rd episode\n', episodes[2])
    V = {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0}
    N = {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0}
    MC(episodes, V, N, gamma)
    print('each value by MC is')
    print(V)

    timestep_max = 1000
    episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
    episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)

    rho_1 = occupancy(episodes_1, 's4', 'prob_to', timestep_max, gamma)
    rho_2 = occupancy(episodes_2, 's4', 'prob_to', timestep_max, gamma)
    print(rho_1, rho_2)