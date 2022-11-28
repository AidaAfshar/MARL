from random import randrange

import numpy as np
#import seaborn as sns
import matplotlib.pylab as plt

from Algorithm.SARSA import SARSA
from MASystem.learners.learner import learner
from MASystem.world.grid_world import grid_world
from MASystem.teacher.oracle import oracle

# we look at mean reward in last window
def grid_search(alpha_lower_bound, alpha_upper_bound, alpha_division_step, gamma_lower_bound, gamma_upper_bound, gamma_division_step):
    alphas = generate_step_list(alpha_lower_bound, alpha_upper_bound, alpha_division_step)
    gammas = generate_step_list(gamma_lower_bound, gamma_upper_bound, gamma_division_step)
    rewards = []
    for alpha in alphas:
        sum_rewards = []
        for gamma in gammas:
            s = get_sum_reward(alpha, gamma)
            sum_rewards.append(s)
        rewards.append(sum_rewards)
    return rewards


def generate_step_list(lower_bound, upper_bound, step):
    x_list = []
    x = lower_bound
    while x <= upper_bound:
        x_list.append(x)
        x += step
    return x_list


def setup_random_apples(apples_num, width, height):
    apples_loc = []
    for i in range(apples_num):
        x = randrange(width)
        y = randrange(height)
        apples_loc.append([x, y])
    return apples_loc


def get_sum_reward(alpha, gamma):
    width = 20
    height = 20
    apples_num = 50

    apples_loc = setup_random_apples(apples_num, width, height)
    world = grid_world(width, height, apples_num, apples_loc)
    orc = oracle(grid_world)
    learner0 = learner('0', 1, 1, world, orc)
    learner1 = learner('1', 8, 8, world, orc)
    learners = [learner0, learner1]

    total_episodes = 1000
    episode_length = 100
    epsilon = 0.9
    model = SARSA(world, learners, orc, total_episodes, episode_length, alpha, gamma, epsilon)
    model.train()

    mean_rew0 = model.get_last_window_mean_reward(0, 100)
    mean_rew1 = model.get_last_window_mean_reward(0, 100)

    return mean_rew0+mean_rew1


if __name__ == '__main__':

    rewards = grid_search(0, 1, 0.1, 0, 1, 0.1)
    print(rewards)
    plt.xlabel("gamma")
    plt.ylabel("alpha")
    plt.imshow(rewards, cmap='hot', interpolation='nearest')
    plt.savefig('grid search heatmap')