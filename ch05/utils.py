def print_agent(agent, env, action_meaning=('^', 'v', '<', '>'), disastor=tuple(range(37, 47)), end=(47,)):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disastor:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()