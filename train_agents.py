import torch
import gymnasium as gym
from itertools import count
import random
import math
from copy import deepcopy
from collections import namedtuple
import os
import numpy as np

from training_utils import get_config, ReplayBuffer, optimize_model_deep_q_learning, save_config_and_model
from plotting_utils import plot_durations_rewards
from models import DQN

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def train_agent_dq_learning(config, render=False):
    if torch.cuda.is_available():
        device = "cuda"
    #elif torch.backends.mps.is_available():
    #    device = "mps"
    else:
        device = "cpu"
    device = torch.device(device)
    config['device'] = device
    env = gym.make(config['environment']['name'])
    buffer = ReplayBuffer(config['environment']['buffer_size'], config['training']['batch_size'])

    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy_model = DQN(n_observations, n_actions, config['model']['n_hidden']).to(device)
    target_model = deepcopy(policy_model).to(device)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config['optimizer']['lr'], amsgrad=True)
    episode_durations, episode_rewards = [], []

    steps_done = 0
    for i_episode in range(config['training']['n_episodes']):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        step_rewards = []
        for t in count():
            sample = random.random()
            eps_threshold = config['training']['epsilon_end'] + (config['training']['epsilon_start'] - config['training']['epsilon_end']) * math.exp(
                -1. * steps_done / config['training']['epsilon_decay'])
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    action = policy_model(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward + abs(observation[1]) + observation[0]**2], device=device, dtype=torch.float32) # make generic reward function call
            step_rewards.append(reward.cpu())

            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            buffer.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model_deep_q_learning(policy_model, target_model, buffer, optimizer, device, config['training']['gamma'])

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_model_state_dict = target_model.state_dict()
            policy_model_state_dict = policy_model.state_dict()
            for key in policy_model_state_dict:
                target_model_state_dict[key] = policy_model_state_dict[key] * config['optimizer']['tau'] + target_model_state_dict[key] * (1 - config['optimizer']['tau'])
            target_model.load_state_dict(target_model_state_dict)

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(np.mean(step_rewards))
                break
        if (i_episode + 1) % 10 == 0:
            print(f"Episode {i_episode + 1} complete.")
    save_config_and_model(episode_durations, episode_rewards, config, policy_model, optimizer, save_model=True)
    if render:
        render_policy_model(config, policy_model, hidden_size=config['model']['n_hidden'])
    return policy_model


def render_policy_model(config, policy_model=None, path=None, hidden_size=None):
    assert policy_model is not None or path is not None, "Must specificy model or model path for loading."
    device = config['device']
    env = gym.make(config['environment']['name'], render_mode="human")
    env.reset()
    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    if path:
        policy_model = DQN(n_observations, n_actions, hidden_size).to(device)
        policy_model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=device))
        policy_model.eval()

    done = False
    total_reward = 0

    while not done:
        env.render()
        state = torch.tensor(env.state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = policy_model(state).max(1)[1].view(1, 1)

        _, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        done = terminated or truncated

    print(f"Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    cp_config = get_config("config_files/mountaincar_dql.yml")
    trained_policy_model = train_agent_dq_learning(cp_config, True)
    #exp_number = "062823162425721572"
    #render_policy_model(cp_config, path=os.path.join(cp_config['data_path'], exp_number), hidden_size=1000)
