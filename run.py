import argparse
import enum
import logging
import time
import pyautogui
from solver import Solver

def benchmark_method(games):
    logging.basicConfig(level=logging.DEBUG)
    for game in range(games):
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

        if not game == games - 1:
            solver.restart()

def victory_method(victories):
    victory_counter = 0
    while victory_counter != victories:
        global_start = time.time()
        solver = Solver()
        result = solver.run()
        final = time.time() - global_start

        if result:
            message = f"Victory in {final}\n"
            victory_counter += 1
        else:
            message = f"Defeat in {final}\n"

        with open("solver1.csv", "a+") as f:
            f.write(f"{result},{final}\n")

        print(message)

        if not victory_counter == victories:
            solver.restart()

def classic_method(games):
    for game in range(games):
        solver = Solver()
        result = solver.run()

        if result:
            print("Victory!")
        else:
            print("Defeat!")

        if not game == games - 1:
            solver.restart()

mapping = [classic_method, benchmark_method, victory_method]

class CMDGamemode(enum.Enum):
    classic     = 0
    benchmark   = 1
    victory     = 2

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s):
        try:
            return CMDGamemode[s]
        except KeyError:
            raise ValueError()

parser = argparse.ArgumentParser(description='Solve a minesweeper grid')
parser.add_argument(
    '-g',
    dest="gamemode",
    type=CMDGamemode.from_string, 
    choices=list(CMDGamemode),
    default=CMDGamemode.classic,
    action="store",
    metavar="GAMEMODE",
    help='Pick from three different gamemode based on whatever you feel like'
)
parser.add_argument(
    '-n',
    dest="runs",
    type=int,
    default=1,
    action="store",
    metavar="GAMEMODE",
    help="How many times you want the game to run the gamemode."
)

args = parser.parse_args()

pyautogui.hotkey("alt", "tab")
try:
    mapping[args.gamemode.value](args.runs)
except Exception:
    pyautogui.hotkey("alt", "tab")
    raise
