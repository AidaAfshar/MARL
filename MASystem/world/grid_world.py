import math
from MASystem.bonus.apple import apple


def get_distance(apple, agent_x, agent_y):
    apple_x, apple_y = apple.get_coordinates()
    delta_x = agent_x - apple_x
    delta_y = agent_y - apple_y
    return math.sqrt(delta_x ** 2 + delta_y ** 2)


class grid_world:

    def __init__(self, width, height, apples_num, apples_initial_loc):
        self.width = width
        self.height = height
        self.apples_num = apples_num
        self.apples_initial_loc = apples_initial_loc
        self.apples = self.initialize_apples()

    def initialize_apples(self):
        apples = []
        for i in range(self.apples_num):
            app = apple(self.apples_initial_loc[i][0], self.apples_initial_loc[i][1])
            apples.append(app)
        return apples

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def has_apple_at(self, x, y):
        for apple in self.apples:
            if not apple.is_eaten():
                if (apple.get_coordinates()) == (x, y):
                    return True, apple.get_value()
        return False, 0

    def no_apple_left(self):
        for apple in self.apples:
            if not apple.is_eaten():
                return False
        return True

    def eat_apple_at(self, x, y):
        for apple in self.apples:
            if (apple.get_coordinates()) == (x, y) and (not apple.is_eaten()):
                apple.set_eaten(True)
                return apple.get_value()
        return 0

    def find_closest_apple(self, agent_x, agent_y):
        closest_apple = None
        min_distance = math.inf
        for apple in self.apples:
            if not apple.is_eaten():
                d = get_distance(apple, agent_x, agent_y)
                if d < min_distance:
                    min_distance = d
                    closest_apple = apple
        return closest_apple

    def reset(self):
        self.apples = self.initialize_apples()

    def print_world(self):
        for i in range(self.width):
            for j in range(self.height):
                b, v = self.has_apple_at(i, j)
                if b:
                    print(v, end="   ")
                else:
                    print(0, end="   ")
            print()
        print("------------------------")



