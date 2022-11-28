from Algorithm.Q_learning import Q_learning
from Algorithm.SARSA import SARSA
from MASystem.learners.learner import learner
from MASystem.world.grid_world import grid_world
from MASystem.teacher.oracle import oracle
from MASystem.plotting.reward_plot import plot_acc_reward
from MASystem.plotting.reward_plot import plot_acc_reward_per_window
from random import randrange

from random import randrange

if __name__ == '__main__':

    width = 30
    height = 30
    apples_num = 1
    apples_loc = [[randrange(width), randrange(width)]]

    grid_world = grid_world(width, height, apples_num, apples_loc)
    grid_world.print_world()
    oracle = oracle(grid_world)
    learner0 = learner('0', randrange(width), randrange(width), grid_world, oracle)
    learner1 = learner('1', randrange(width), randrange(width), grid_world, oracle)
    learners = [learner0, learner1]
    print("learner 0 initial state : ", learner0.get_current_state())
    print("learner 1 initial state : ", learner1.get_current_state())
    print("-----------------------------------------")


    # RL algorithm
    total_episodes = 2000
    episode_length = 800
    alpha = 0.1  # learning rate
    gamma = 1    # discount factor
    epsilon = 0.3
    model = Q_learning(grid_world, learners, oracle, total_episodes, episode_length, alpha, gamma, epsilon)
    # model = SARSA(grid_world, learners, oracle, total_episodes, episode_length, alpha, gamma, epsilon)
    model.train()
    # save ?

    reward0 = model.get_acc_reward_for_learner(learner0)
    reward1 = model.get_acc_reward_for_learner(learner1)
    plot_acc_reward(reward0, 'learner0 acc reward per episode')
    plot_acc_reward(reward1, 'learner1 acc reward per episode')

    sum_rew = []
    for i in range(len(reward0)):
        r = reward0[i] + reward1[i]
        sum_rew.append(r)

    print(sum_rew)
    plot_acc_reward_per_window(sum_rew, 2, 'sum of learners acc reward per episode')

    model.print_Q_table(learner0)
    model.print_Q_table(learner1)
