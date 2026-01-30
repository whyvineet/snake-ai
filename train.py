import os
from game import SnakeGame
from agent import Agent, get_state
from helper import plot


def train(render=True, plot_every=1):
    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    model_path = "models/model.pth"
    if os.path.exists(model_path):
        agent.model.load(model_path, device=agent.device)
        print("Loaded existing model.")
    game = SnakeGame(render=render)

    while True:
        state_old = get_state(game)
        action = agent.get_action(state_old)

        reward, done, score = game.play_step(action)
        state_new = get_state(game)

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)

            if score > record:
                record = score
                agent.model.save()
                print("Model saved!")

            if plot_every and agent.n_games % plot_every == 0:
                plot(scores, mean_scores)
            print(
                f"Game {agent.n_games} | "
                f"Score {score} | "
                f"Mean {mean_score:.2f}"
            )


if __name__ == "__main__":
    train(render=True, plot_every=1)
