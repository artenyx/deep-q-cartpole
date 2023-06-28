import torch
import matplotlib.pyplot as plt
import os


def plot_durations_rewards(durations, rewards, path):
    plt.figure(1)
    #durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations)
    plt.savefig(os.path.join(path, "durations.png"))
    plt.close()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards)
    plt.savefig(os.path.join(path, "rewards.png"))
    plt.close()



