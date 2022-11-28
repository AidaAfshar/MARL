import gym
import numpy as np


class learner(gym.Env):

    def __init__(self, id, initial_x, initial_y, grid_world, oracle, max_episode_steps=100, reward_advice_window=10):
        super().__init__()
        self.id = id
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.x = initial_x
        self.y = initial_y
        self.grid_world = grid_world
        self.oracle = oracle
        self.max_episode_steps = max_episode_steps
        self.reward_advice_window = reward_advice_window
        self.acc_reward_in_advice_window = 0
        self.step_count = 0

        # Open AI Gym generic
        self.possible_actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']
        self.action_space = gym.spaces.Discrete(4)  # 0 ~ UP | 1 ~ DOWN | 2 ~ RIGHT | 3 ~ LEFT
        self.observation_space = gym.spaces.MultiDiscrete([grid_world.get_width(), grid_world.get_height()])
        self.state = (initial_x, initial_y)

    def step(self, action):
        self.step_count += 1

        # Execute the action
        if action == 'UP':
            self.move_up()
        elif action == 'DOWN':
            self.move_down()
        elif action == 'RIGHT':
            self.move_right()
        elif action == 'LEFT':
            self.move_left()


        # check if there exists an apple on this tile. If yes, eat it!
        b, _ = self.grid_world.has_apple_at(self.x, self.y)
        if b:
            apple_reward = self.grid_world.eat_apple_at(self.x, self.y)
        else:
            apple_reward = -1

        # State
        self.state = (self.x, self.y)
        observation = np.array([self.x, self.y])

        # Done
        if self.step_count > self.max_episode_steps:
            done = True
        else:
            done = self.grid_world.no_apple_left()

        # Reward
        reward = apple_reward

        return observation, reward, done, {}

    def reset(self):
        self.step_count = 0
        x = self.x
        y = self.y
        self.x = self.initial_x
        self.y = self.initial_y
        self.state = (self.initial_x, self.initial_y)
        return np.array([x, y])

    def move_up(self):
        if self.y > 0:
            # print("moving to up")
            self.y = self.y - 1

    def move_down(self):
        if self.y < self.grid_world.get_height() - 1:
            # print("moving to down")
            self.y = self.y + 1

    def move_right(self):
        if self.x < self.grid_world.get_width() - 1:
            # print("moving to right")
            self.x = self.x + 1

    def move_left(self):
        if self.x > 0:
            # print("moving to left")
            self.x = self.x - 1

    def needs_advice(self):
        # TODO
        if ((self.step_count % self.reward_advice_window) == 0) and (self.acc_reward_in_advice_window == 0):
            self.acc_reward_in_advice_window = 0
            return True
        else:
            return False

    def get_current_state(self):
        return self.state

    def get_id(self):
        return self.id

    def get_possible_actions(self):
        return self.possible_actions
