import matplotlib.pyplot as plt


def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])

    plt.xlabel('time steps')
    plt.ylabel('cumulative regrets')
    plt.title(f'{solvers[0].bandit.k}-armed bandit')
    plt.legend()
    plt.show()