class apple:

    def __init__(self, x, y):
        self.value = 10
        self.x = x
        self.y = y
        self.eaten = False

    def get_coordinates(self):
        return self.x, self.y

    def is_eaten(self):
        return self.eaten

    def set_eaten(self, boolean):
        self.eaten = boolean

    def get_value(self):
        return self.value

    def print(self):
        print("x : ", self.x)
        print("y : ", self.y)
        print("value : ", self.value)
        print("eaten : ", self.eaten)
        print('---------------------------')
