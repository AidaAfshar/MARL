class oracle:

    def __init__(self, grid_world):
        self.grid_world = grid_world

    def get_advice(self, agent):
        (agent_x, agent_y) = agent.get_current_state()
        closest_apple = self.grid_world.find_closest_apple(agent_x, agent_y)
        closest_apple_x, closest_apple_y = closest_apple.get_coordinates()
        action_advice = self.choose_action(agent_x, agent_y, closest_apple_x, closest_apple_y)
        return action_advice

    def choose_action(self, agent_x, agent_y, closest_apple_x, closest_apple_y):
        # move toward the farther axis of closest apple
        delta_x = agent_x - closest_apple_x
        delta_y = agent_y - closest_apple_y
        if abs(delta_x) > abs(delta_y):
            action = 3 if delta_x > 0 else 1
        else:
            action = 0 if delta_y > 0 else 2
        return action
