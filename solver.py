import cv2
import pyautogui
import numpy as np
import random
from PIL import ImageGrab
import enum
import time

class Categories(enum.Enum):
    bomb =  0
    full =  1
    empty = 2

class Gamemodes(enum.Enum):
    beginner = (300, 476)
    intermediate = (556, 732)
    expert = (1004, 732)

class Dimensions(enum.Enum):
    beginner = (8, 8)
    intermediate = (16, 16)
    expert = (30, 16)
    

class Solver:
    def __init__(self):
        self.grid_top_left = (26, 202)
        self.cube_size = (32, 32)

        self.number_colors = [
            (0, 0, 255), (0, 128, 0), 
            (255, 0, 0), (0, 0, 128), 
            (128, 0, 0), (0, 128, 128), 
            (0, 0, 0), (128, 128, 128)
        ]

        self.categories = {
            "bomb": cv2.imread("assets/flag.png"),
            "full": cv2.imread("assets/full.png"),
            "empty": cv2.imread("assets/empty.png")
        }

        self.template_threshold = 0.95

        self.playing = cv2.imread("assets/playing.png")
        self.win = cv2.imread("assets/win.png")

        self.box = pyautogui.locateOnScreen("assets/top_left.png")
        self.game = None
        self.coords = []
        self.states = []

    @staticmethod
    def take_screenshot():
        pyautogui.hotkey("alt", "printscreen")
        sc = ImageGrab.grabclipboard()
        game = cv2.cvtColor(np.array(sc), cv2.COLOR_RGB2BGR)

        return game

    def get_updated_cubes(self):
        start = time.time()

        self.coords = []
        self.states = []

        self.game = self.take_screenshot()
        dimensions = (self.game.shape[1], self.game.shape[0])
        self.mode = Gamemodes(dimensions)

        for category in Categories:
            template = cv2.matchTemplate(self.game, self.categories[category.name], cv2.TM_CCOEFF_NORMED)
            loc = np.where( template >= self.template_threshold)

            for pt in zip(*loc[::-1]):
                self.coords.append(pt)
                self.states.append(category)

        for dim_y in range(Dimensions[self.mode.name].value[1]):
            y = self.grid_top_left[1] + (32*dim_y)

            for dim_x in range(Dimensions[self.mode.name].value[0]):
                x = self.grid_top_left[0] + (32*dim_x)

                if (x, y) not in self.coords:
                    b, g, r = self.game[y + 16, x + 16]
                    number = self.number_colors.index((r, g, b)) + 1

                    self.coords.append((x, y))
                    self.states.append(number)


        self.coords, self.states = (list(t) for t in zip(*sorted(zip(self.coords, self.states), key=lambda k: [k[0][1], k[0][0]])))
        print(f"Update Cubes Time Taken: {time.time() - start}")

    def is_flag(self, x, y):
        flag = cv2.matchTemplate(self.game, self.categories["bomb"], cv2.TM_CCOEFF_NORMED)
        loc = list(zip(*np.where( flag >= self.template_threshold)[::-1]))

        is_flag = self.states[self.coords.index((x, y))] == Categories.bomb

        return (x, y) in loc or is_flag

    def click_cube(self, x, y, *, click="left"):
        if self.is_flag(x, y):
            return

        pyautogui.click(self.box.left + x + 16, self.box.top + y + 16, button=click)    

    def is_playing(self):
        playing = cv2.matchTemplate(self.game, self.playing, cv2.TM_CCOEFF_NORMED)
        loc = list(zip(*np.where( playing >= self.template_threshold)[::-1]))
        return bool(loc)

    def result(self):
        playing = cv2.matchTemplate(self.game, self.win, cv2.TM_CCOEFF_NORMED)
        loc = list(zip(*np.where( playing >= self.template_threshold)[::-1]))
        return bool(loc)  

    def get_neighbors(self, index):
        start = time.time()
        dimension = Dimensions[self.mode.name]
        row_len = dimension.value[0]
        col_len = dimension.value[1]

        most_left = index % row_len == 0
        most_right = index % row_len == row_len - 1

        neighbors_index = [index - row_len, index, index + row_len]

        if not most_right:
            neighbors_index.extend([index + 1 - row_len, index + 1, index + 1 + row_len])

        if not most_left:
            neighbors_index.extend([index - 1 - row_len, index - 1, index - 1 + row_len])

        filtered_neighbors = [neighbor for neighbor in neighbors_index if 0 < neighbor < (row_len * col_len)]

        neighbors = []
        for neighbor in filtered_neighbors:
            try:
                if self.states[neighbor] in [Categories.bomb, Categories.full]:
                    neighbors.append(neighbor)
            except IndexError:
                pass

        print(f"Neighbors Time Taken: {time.time() - start}")
        return neighbors

    def make_decision(self, neighbors, index, change):
        start = time.time()

        full_or_bomb_neighbors = [self.states[neighbor] for neighbor in neighbors if self.states[neighbor] in [Categories.full, Categories.bomb]]
        if len(full_or_bomb_neighbors) == self.states[index] and not all([neighbor == Categories.bomb for neighbor in full_or_bomb_neighbors]):
            print("Marking a bomb")
            for bomb in neighbors:
                if self.states[bomb] == Categories.bomb:
                    continue

                self.click_cube(*self.coords[bomb], click="right")
                self.states[bomb] = Categories.bomb
                change = True
                
            self.get_updated_cubes()

        bomb_neighbors = [self.states[neighbor] for neighbor in neighbors if self.states[neighbor] == Categories.bomb]
        if len(bomb_neighbors) == self.states[index] and any([self.states[neighbor] == Categories.full for neighbor in neighbors]):
            print("Clicking on some cubes")
            for to_click in neighbors:
                if self.states[to_click] == Categories.full:
                    self.click_cube(*self.coords[to_click])
                    change = True
                    
            self.get_updated_cubes()

        print(f"Make Decision Time Taken: {time.time()-start}")
        return change

    def solve(self):
        self.get_updated_cubes()

        while True:
            change = False
            for index, _ in enumerate(self.states):
                if not isinstance(self.states[index], int):
                    continue

                if not self.is_playing():
                    return self.result()

                neighbors = self.get_neighbors(index)
                # print(f"Number: {self.states[index]}")
                # print(f"Neighbors: {neighbors}")

                change = self.make_decision(neighbors, index, change) 

            if not change:
                print("Out of options, picking a random cube")
                coord = random.choice([coord for coord in self.coords if self.states[self.coords.index(coord)] == Categories.full])
                self.click_cube(*coord) 
                self.get_updated_cubes()


if __name__ == '__main__':
    pyautogui.hotkey("alt", "tab")

    global_start = time.time()
    solver = Solver()
    result = solver.solve()

    if result:
        print(f"Victory in {time.time() - global_start}")
    else:
        print("Defeat")
