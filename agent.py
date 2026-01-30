import random
import numpy as np
from collections import deque
import torch

from game import Direction, Point, BLOCK_SIZE
from model import LinearQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = LinearQNet(11, 256, 3).to(self.device)
        self.trainer = QTrainer(self.model, LR, self.gamma, self.device)

    def get_action(self, state):
        self.epsilon = max(0, 80 - self.n_games)
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            move = torch.argmax(self.model(state0)).item()

        final_move[move] = 1
        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        mini_sample = (
            random.sample(self.memory, BATCH_SIZE)
            if len(self.memory) > BATCH_SIZE
            else self.memory
        )
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


def get_state(game):
    head = game.head

    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        (dir_r and game._is_collision(point_r)) or
        (dir_l and game._is_collision(point_l)) or
        (dir_u and game._is_collision(point_u)) or
        (dir_d and game._is_collision(point_d)),

        (dir_u and game._is_collision(point_r)) or
        (dir_d and game._is_collision(point_l)) or
        (dir_l and game._is_collision(point_u)) or
        (dir_r and game._is_collision(point_d)),

        (dir_d and game._is_collision(point_r)) or
        (dir_u and game._is_collision(point_l)) or
        (dir_r and game._is_collision(point_u)) or
        (dir_l and game._is_collision(point_d)),

        dir_l, dir_r, dir_u, dir_d,
        game.food.x < game.head.x,
        game.food.x > game.head.x,
        game.food.y < game.head.y,
        game.food.y > game.head.y,
    ]

    return np.array(state, dtype=int)
