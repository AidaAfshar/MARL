from Algorithm.Q_learning import Q_learning
from Algorithm.SARSA import SARSA
from MASystem.learners.learner import learner
from MASystem.world.grid_world import grid_world
from MASystem.teacher.oracle import oracle
from MASystem.plotting.reward_plot import plot_acc_reward
from MASystem.plotting.reward_plot import plot_acc_reward_per_window

from random import randrange

if __name__ == '__main__':

    width = 30
    height = 30
    apples_num = 1
    # apples_loc = [[1, 0], [0, 2], [0, 3], [3, 0], [2, 2], [3, 3]]
    apples_loc = [[randrange(width), randrange(width)]]

    '''
    apples_loc = []
    for i in range(apples_num):
        x = randrange(width)
        y = randrange(height)
        apples_loc.append([x, y])
    print(apples_loc)
    '''

    grid_world = grid_world(width, height, apples_num, apples_loc)
    grid_world.print_world()
    oracle = oracle(grid_world)
    learner0 = learner('0', randrange(width), randrange(width), grid_world, oracle)
    learners = [learner0]
    print("learner's initial state : ", learner0.get_current_state())
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
    plot_acc_reward(reward0, 'learner0 acc reward per episode')
    print(reward0)
    # plot_acc_reward_per_window(reward0, 100, 'single - leaner0 mean of _acc reward per episode_ per window')

    model.print_Q_table(learner0)
