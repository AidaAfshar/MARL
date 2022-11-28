import numpy as np


class Q_learning:

    def __init__(self, grid_world, learners, oracle, total_episodes, episode_length, alpha, gamma, epsilon):
        self.grid_world = grid_world
        self.learners = learners
        self.oracle = oracle
        self.total_episodes = total_episodes
        self.episode_length = episode_length
        self.alpha = alpha  # learning rate
        self.gamma = gamma
        self.epsilon = epsilon  # epsilon greedy
        self.Q_tables = self.initialize_Q_tables()
        self.rewards = self.initialize_rewards()


    def initialize_Q_tables(self):
        """
        Q_tables = {learner '0' : {(0, 0): {'UP': 0, 'DOWN': 0, 'RIGHT': 0, 'LEFT': 0}},
                                   (0, 1): {'UP': 0, 'DOWN': 0, 'RIGHT': 0, 'LEFT': 0}},
                                   ...},
                    learner '1' : {(0, 0): {'UP': 0, 'DOWN': 0, 'RIGHT': 0, 'LEFT': 0}},
                                   (0, 1): {'UP': 0, 'DOWN': 0, 'RIGHT': 0, 'LEFT': 0}},
                                   ...
                    }
        """
        Q_tables = dict()   # dictionary of dictionaries
        for learner in self.learners:
            Q_tables[learner.get_id()] = dict()
            for i in range(self.grid_world.get_width()):
                for j in range(self.grid_world.get_height()):
                    Q_tables[learner.get_id()][(i, j)] = {'UP': 0, 'DOWN': 0, 'RIGHT': 0, 'LEFT': 0}
        return Q_tables


    def initialize_rewards(self):
        """
        rewards = {learner '0' : { episode 0 : [r0, r1, ... ],
                                   episode 1 : [r0, r1, ... ],
                                   ...
                                   episode T : [r0, r1, ... ] },

                   learner '1' : { episode 0 : [r0, r1, ... ],
                                   episode 1 : [r0, r1, ... ],
                                   ...
                                   episode T : [r0, r1, ... ] },

                   ... }

        """
        rewards = dict()
        for learner in self.learners:
            rewards[learner.get_id()] = dict()
            for e in range(self.total_episodes):
                rewards[learner.get_id()][e] = []
        return rewards

    def choose_action(self, learner, state):    #epsilon-greedy
        if np.random.uniform(0, 1) < self.epsilon:
            action_number = learner.action_space.sample()
            action = learner.get_possible_actions()[action_number]
        else:
            Q_table = self.Q_tables[learner.get_id()]
            q = Q_table[state]   # q = {'UP': 0, 'DOWN': 0.1, 'RIGHT': 2, 'LEFT': -0.3}}
            max_q_value = max(q.values())
            action = np.random.choice([i for i, v in q.items() if v == max_q_value])
        return action

    def update_Q_value(self, learner, state1, action1, state2, reward):
        """
        Q-learning update rule :
        Q(s,a) <- Q(s,a) + alpha [reward + gamma * max_a Q(s',a)  - Q(s,a)]
        """
        Q_table = self.Q_tables[learner.get_id()]
        current_q_value = Q_table[state1][action1]    # Q(s,a)
        max_q_value_of_next_state = max(Q_table[state2].values())      # max_a Q(s',a)
        new_q_value = current_q_value + self.alpha*(reward + (self.gamma*max_q_value_of_next_state) - current_q_value)
        self.Q_tables[learner.get_id()][state1][action1] = new_q_value

    def train(self):
        for episode in range(self.total_episodes):
            print('------------------------------')
            print('episode : ', episode)
            for t in range(self.episode_length):
                print("------------------")
                print('time step : ', t)
                for learner in self.learners:
                    state1 = learner.get_current_state()
                    print('state : ', state1)
                    action1 = self.choose_action(learner, state1)
                    print('action : ', action1)
                    obs, reward, done, info = learner.step(action1)
                    print('reward : ', reward)
                    state2 = (obs[0], obs[1])
                    # state2 = learner.get_current_state()
                    print('new state : ', state2)
                    self.update_Q_value(learner, state1, action1, state2, reward)
                    self.rewards[learner.get_id()][episode].append(reward)
                    if done:
                        break

                if done:
                    break
            self.reset()
        print('--------------------------------------')

    def reset(self):
        for learner in self.learners:
            learner.reset()
        self.grid_world.reset()

    def get_Q_tables(self):
        return self.Q_tables

    def get_Q(self, learner):
        return self.Q_tables[learner.get_id()]

    def get_acc_reward_for_learner(self, learner):
        episode_rewards = self.rewards[learner.get_id()]
        acc_episode_rewards = []
        for e in range(self.total_episodes):
            rew_list = episode_rewards[e]
            s = sum(rew_list)
            acc_episode_rewards.append(s)
        return acc_episode_rewards  # returns [R1, R2, ..., RT] ; T = total_episodes ; Ri = sum of rewards in each time step

    def print_Q_table(self, learner):
        print("------------------------------------------------")
        print("Q table for learner", learner.get_id())
        Q_table = self.Q_tables[learner.get_id()]
        for key, value in Q_table.items():
            print(key, " : ", value)
        print("------------------------------------------------")




    '''
    def get_last_window_mean_reward(self, learner_index, window_length):
        l1 = self.get_episode_rewards_list(learner_index)
        l1 = l1[(-1)*window_length:]
        import statistics
        mean_reward = statistics.mean(l1)
        return mean_reward

    '''