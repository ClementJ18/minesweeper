import cv2
import pyautogui
import numpy as np
import random
from PIL import ImageGrab
import enum
import time
import sys
import logging

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

        self.number_coords = [(38, 126), (64, 126), (90, 126)]
        self.number_template_index = [
            (0, 0), (26, 0), (52, 0), (78, 0), (104, 0), 
            (130, 0), (156, 0), (182, 0), (208, 0), (234, 0)
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
        self.number_template = cv2.imread("assets/numbers.png")


        self.game = None
        self.coords = []
        self.states = []
        self.solved = []
        self.mines_current = 0
        self.mines_max = 0

    @staticmethod
    def take_screenshot():
        pyautogui.hotkey("alt", "printscreen")
        sc = ImageGrab.grabclipboard()
        game_img = cv2.cvtColor(np.array(sc), cv2.COLOR_RGB2BGR)

        return game_img

    @staticmethod
    def restart():
        for x in ["win", "dead", "playing"]:
            re = pyautogui.locateOnScreen(f"assets/{x}.png")

            if not re is None:
                break
        else:
            return

        pyautogui.click(re.left + 16, re.top + 16, button="left")

    def get_mines_amount(self):
        number = ''
        for x, y in self.number_coords:
            match = cv2.matchTemplate(self.number_template, self.game[y:y+41, x:x+21], cv2.TM_CCOEFF_NORMED)
            loc = list(zip(*np.where( match >= 0.95)[::-1]))

            number += str(self.number_template_index.index(loc[0]))

        return int(number)

    #timeME
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
            matches = list(zip(*loc[::-1]))

            self.coords.extend(matches)
            self.states.extend([category] * len(matches))

        for dim_y in range(Dimensions[self.mode.name].value[1]):
            y = self.grid_top_left[1] + (32*dim_y)

            for dim_x in range(Dimensions[self.mode.name].value[0]):
                x = self.grid_top_left[0] + (32*dim_x)

                if (x, y) not in self.coords:
                    b, g, r = self.game[y + 16, x + 16]
                    number = self.number_colors.index((r, g, b)) + 1

                    self.coords.append((x, y))
                    self.states.append(number)

        self.mines_current = self.get_mines_amount()

        self.coords, self.states = (list(t) for t in zip(*sorted(zip(self.coords, self.states), key=lambda k: [k[0][1], k[0][0]])))
        
        if self.states[-1] != Categories.full:
            self.states.reverse()
            self.coords.reverse()

        time_taken = time.time() - start
        logging.info(f"Update Cubes Time Taken: %s", time_taken)
        return time_taken

    def click_cube(self, x, y, *, click="left"):
        flag = cv2.matchTemplate(self.game, self.categories["bomb"], cv2.TM_CCOEFF_NORMED)
        loc = list(zip(*np.where( flag >= self.template_threshold)[::-1]))

        is_bomb = self.states[self.coords.index((x, y))] == Categories.bomb
        if (x, y) in loc or is_bomb:
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

    def to_matrix(self):
        dim = Dimensions[self.mode.name]
        row_len = dim.value[0]
        matrix_coords = [self.coords[y*row_len:(y+1)*row_len] for y in range(dim.value[1])]
        matrix_states = [self.states[y*row_len:(y+1)*row_len] for y in range(dim.value[1])
]
        return matrix_coords, matrix_states

    #timeME
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

        neighbors = [neighbor for neighbor in neighbors_index if 0 < neighbor < (row_len * col_len) and self.states[neighbor] in [Categories.bomb, Categories.full]]

        logging.info(f"Neighbors Time Taken: %s", (time.time() - start))
        return neighbors

    #timeME
    def make_a_simple_decision(self, neighbors, index, change):
        start = time.time()
        decision_time = 0

        full_or_bomb_neighbors = [self.states[neighbor] for neighbor in neighbors if self.states[neighbor] in [Categories.full, Categories.bomb]]
        if len(full_or_bomb_neighbors) == self.states[index] and not all([neighbor == Categories.bomb for neighbor in full_or_bomb_neighbors]):
            logging.debug("Marking a bomb")
            for bomb in neighbors:
                if self.states[bomb] == Categories.bomb:
                    continue

                self.click_cube(*self.coords[bomb], click="right")
                self.states[bomb] = Categories.bomb
                change = True
                
                decision_time += self.get_updated_cubes()

        bomb_neighbors = [self.states[neighbor] for neighbor in neighbors if self.states[neighbor] == Categories.bomb]
        if len(bomb_neighbors) == self.states[index] and any([self.states[neighbor] == Categories.full for neighbor in neighbors]):
            logging.debug("Clicking on some cubes")
            for to_click in neighbors:
                if self.states[to_click] == Categories.full:
                    self.click_cube(*self.coords[to_click])
                    change = True
                    
                decision_time += self.get_updated_cubes()

        logging.info(f"Make Decision Time Taken: %s", (time.time() - start - decision_time))
        return change

    def run(self):
        self.get_updated_cubes()
        self.mines_max = self.get_mines_amount()

        while self.is_playing():
            self.solve()

        return self.result()

    def solve(self):
        change = False

        for index, state in enumerate(self.states):
            if not isinstance(state, int):
                continue

            if index in self.solved:
                continue

            logging.info("Resolving for %s @ %s", state, index)

            neighbors = self.get_neighbors(index)
            if not any([self.states[neighbor] == Categories.full for neighbor in neighbors]):
                self.solved.append(index)
                continue

            change = self.make_a_simple_decision(neighbors, index, change)

        if not change:
            logging.info("Out of options, picking a random cube")
            coord = random.choice([coord for coord in self.coords if self.states[self.coords.index(coord)] == Categories.full])
            self.click_cube(*coord) 
            self.get_updated_cubes()

if __name__ == '__main__':
    pyautogui.hotkey("alt", "tab")
    logging.basicConfig(level=logging.DEBUG)
    global_start = time.time()
    solver = Solver()
    result = solver.run()
    final = time.time() - global_start

    if result:
        message = f"Victory in {final}\n"
    else:
        message = f"Defeat in {final}\n"

    with open("solver1.csv", "a+") as f:
        f.write(f"{result},{final}\n")

    logging.debug(message)
