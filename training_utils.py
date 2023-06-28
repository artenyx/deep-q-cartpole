import torch
import torch.nn as nn
from dataclasses import dataclass
from collections import deque, namedtuple
import random
import yaml
from datetime import datetime
import os

from plotting_utils import plot_durations_rewards


@dataclass
class Transition_data:
    state: torch.tensor
    action: torch.tensor
    next_state: torch.tensor
    reward: torch.tensor


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):
    def __init__(self,
                 capacity: int,
                 batch_size: int):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def push(self,
             state: torch.tensor,
             action: torch.tensor,
             next_state: torch.tensor,
             reward: torch.tensor):
        """Save a transition"""
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def full(self):
        return len(self) == self.capacity

    def full_batch(self):
        return len(self) >= self.batch_size

    def generate_batches(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        transitions = self.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        return batch

    def __len__(self):
        return len(self.memory)


def get_config(config_path="config.yml"):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    return config


def optimize_model_deep_q_learning(policy_model,
                                   target_model,
                                   buffer,
                                   optimizer,
                                   device,
                                   gamma=0.99):
    if not buffer.full_batch():
        return
    batch = buffer.generate_batches()

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    batched_states, batched_actions, batched_rewards = (torch.cat(batch.state),
                                                        torch.cat(batch.action),
                                                        torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_model(batched_states).gather(1, batched_actions)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(buffer.batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + batched_rewards

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_model.parameters(), 100)
    optimizer.step()


def save_config_and_model(episode_durations, episode_rewards, config, policy_model=None, optimizer=None, save_model=False):
    experiment_dir = os.path.join(config['data_path'], datetime.now().strftime("%m%d%y%H%M%S%f"))

    # Create the experiment directory
    os.makedirs(experiment_dir)

    # Save the config as YAML
    config_file_path = os.path.join(experiment_dir, config['environment']['name'] + '_dql.yml')
    with open(config_file_path, 'w') as file:
        yaml.dump(config, file)

    # Save the PyTorch model and optimizer state if specified
    if save_model and policy_model is not None:
        model_file_path = os.path.join(experiment_dir, 'model.pt')
        torch.save(policy_model.state_dict(), model_file_path)

        if optimizer is not None:
            optimizer_file_path = os.path.join(experiment_dir, 'optimizer.pt')
            torch.save(optimizer.state_dict(), optimizer_file_path)
    plot_durations_rewards(episode_durations, episode_rewards, experiment_dir)
    return experiment_dir
