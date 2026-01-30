import matplotlib.pyplot as plt

plt.ion()


def plot(scores, mean_scores):
    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Games")
    plt.ylabel("Score")
    plt.plot(scores, label="Score", color="#4F8EF7")
    plt.plot(mean_scores, label="Mean", color="#F5A524")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.pause(0.001)
