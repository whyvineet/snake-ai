import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font(None, 28)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x y")

# Colors
WHITE = (245, 245, 245)
MUTED = (160, 160, 160)
RED = (220, 70, 70)
BLUE = (80, 150, 255)
BLUE_DARK = (50, 110, 220)
BLACK = (20, 20, 24)
GRID = (35, 35, 42)

BLOCK_SIZE = 20
SPEED = 40


class SnakeGame:
    def __init__(self, w=640, h=480, render=True):
        self.w = w
        self.h = h
        self.render = render

        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake AI")
        else:
            self.display = None

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # Only handle events if rendering
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # Move snake
        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        # Collision
        if self._is_collision():
            reward = -10
            game_over = True
            return reward, game_over, self.score

        # Food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # UI
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)

        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        if (
            pt.x < 0 or pt.x >= self.w
            or pt.y < 0 or pt.y >= self.h
        ):
            return True

        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        self._draw_grid()

        # Draw snake
        for idx, pt in enumerate(self.snake):
            color = BLUE_DARK if idx == 0 else BLUE
            pygame.draw.rect(
                self.display,
                color,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.display,
                BLACK,
                pygame.Rect(pt.x + 3, pt.y + 3, BLOCK_SIZE - 6, BLOCK_SIZE - 6),
                1,
            )

        # Draw food as circle
        food_center = (
            self.food.x + BLOCK_SIZE // 2,
            self.food.y + BLOCK_SIZE // 2,
        )
        pygame.draw.circle(self.display, RED, food_center, BLOCK_SIZE // 2 - 2)

        pygame.display.set_caption(f"Snake AI | Score: {self.score}")
        pygame.display.flip()

    def _draw_grid(self):
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(
                self.display,
                GRID,
                (x, 0),
                (x, self.h),
            )
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID, (0, y), (self.w, y))

    def _move(self, action):
        # action = [straight, right, left]
        clock_wise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]

        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]:  # straight
            new_dir = clock_wise[idx]
        elif action == [0, 1, 0]:  # right
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # left
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
