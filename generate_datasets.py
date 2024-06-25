import gymnasium as gym
import dill
from d3rlpy.dataset import MDPDataset
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
import pickle
import random
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from os import sys
sys.path.append('/Users/caroline/Desktop/projects/repos/')
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
from four_room.shortest_path import find_all_action_values
from four_room.utils import obs_to_state


gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)


with open('four_room/configs/fourrooms_train_config.pl', 'rb') as f:
        train_config = dill.load(f)


class CNN(BaseFeaturesExtractor):
    """
    This class is specifies the structure of the NN that gets loaded from models/dqn_four_rooms_suboptimal.zip
    """    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.image_normaliser = 1
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
        )

        with torch.no_grad():
            n_flatten = np.prod(self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1:])

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations / self.image_normaliser
        x = self.cnn(observations)
        x = x.flatten(start_dim=1)
        return self.linear(x)

policy_kwargs = dict(
    features_extractor_class=CNN,
    features_extractor_kwargs=dict(features_dim=512),
    normalize_images=False,
    net_arch=[]
)

custom_objects = {
    "policy_kwargs": policy_kwargs,
    "lr_schedule": lambda _: 0.0001,  # Dummy function for lr_schedule
    "exploration_schedule": lambda _: 0.1  # Dummy function for exploration_schedule
}

suboptimal_model = DQN.load("models/dqn_four_rooms_suboptimal", custom_objects=custom_objects)


def init_train_env(seed):
    """
    Initializes the environment according to the training configurations by the four_room environment

    :seed: Seed to be used when initializing environment
    :return: Initialized training environment 
    """

    # env = wrap_env(gym_wrapper(gym.make(('MiniGrid-FourRooms-v1'))))
    gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
    env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                   agent_pos=train_config['agent positions'],
                                   goal_pos=train_config['goal positions'],
                                   doors_pos=train_config['topologies'],
                                   agent_dir=train_config['agent directions'],
                                   render_mode="rgb_array"))
    env.reset(seed=seed)
    return env


def optimal_policy(obs):
    """
    Optimal policy provided by the four_room environment. It finds the shortest path to the goal. 

    :obs: The observation from the environment 
    :return: Index of the optimal action to take based on the observation
    """
    state = obs_to_state(obs)
    q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
    optimal_action = np.argmax(q_values)
    return optimal_action


def generate_expert_dataset(seed=42):
    """
    Generate a dataset according to the optimal policy in d4rl format.

    :seed: Seed used to initialize the environment
    :return: Expert dataset in d4rl format as Numpy array. 
    """
    env = init_train_env(seed)
    dataset = {'observations':[], 'next_observations':[], 'actions':[], 'rewards':[],
                'terminals':[], 'timeouts':[], 'infos':[]}
    for i in range(len(train_config['topologies'])):
        observation, info = env.reset()
        done = False
        while not done:
            action = optimal_policy(observation)
            last_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            dataset['observations'].append(np.array(last_observation).flatten())
            dataset['next_observations'].append(np.array(observation).flatten())
            dataset['actions'].append(np.array([action]))
            dataset['rewards'].append(reward)
            dataset['terminals'].append(terminated)
            dataset['timeouts'].append(truncated)
            dataset['infos'].append(info)

    for key in dataset:
        dataset[key] = np.array(dataset[key])
    return dataset

def generate_mixed_suboptimal_dataset(seed=42):
    """
    Generate a suboptimal dataset. We select the expert action with 0.5 probability, and we
    select a suboptimal action with probability 0.5. The suboptimal action is from a suboptimal 
    DQN model that we have trained and stopped to reach about 50% of the expert performance.
    """
    

    env = init_train_env(seed)
    random.seed(seed)
    dataset = {'observations':[], 'next_observations':[], 'actions':[], 'rewards':[],
                'terminals':[], 'timeouts':[], 'infos':[]}
    for i in range(len(train_config['topologies'])):
        observation, info = env.reset()
        done = False
        while not done:    
            rand = random.randint(0, 1)
            if rand == 0:
                action = optimal_policy(observation)
            else:
                action = suboptimal_model.predict(observation)[0]
            last_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            dataset['observations'].append(np.array(last_observation).flatten())
            dataset['next_observations'].append(np.array(observation).flatten())
            dataset['actions'].append(np.array([action]))
            dataset['rewards'].append(reward)
            dataset['terminals'].append(terminated)
            dataset['timeouts'].append(truncated)
            dataset['infos'].append(info)

    for key in dataset:
        dataset[key] = np.array(dataset[key])
    return dataset

def generate_suboptimal_dataset(seed=42):
    """
    Generate a suboptimal dataset. We select the expert action with 0.5 probability, and we
    select a suboptimal action with probability 0.5. The suboptimal action is from a suboptimal 
    DQN model that we have trained and stopped to reach about 50% of the expert performance.
    """
    

    env = init_train_env(seed)
    random.seed(seed)
    dataset = {'observations':[], 'next_observations':[], 'actions':[], 'rewards':[],
                'terminals':[], 'timeouts':[], 'infos':[]}
    for i in range(len(train_config['topologies'])):
        observation, info = env.reset()
        done = False
        while not done:    
            rand = random.randint(0, 1)
            action = suboptimal_model.predict(observation)[0]
            last_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            dataset['observations'].append(np.array(last_observation).flatten())
            dataset['next_observations'].append(np.array(observation).flatten())
            dataset['actions'].append(np.array([action]))
            dataset['rewards'].append(reward)
            dataset['terminals'].append(terminated)
            dataset['timeouts'].append(truncated)
            dataset['infos'].append(info)

    for key in dataset:
        dataset[key] = np.array(dataset[key])
    return dataset

def generate_random_dataset(seed=42):
    """
    Generate a suboptimal dataset. We select the expert action with 0.5 probability, and we
    select a suboptimal action with probability 0.5. The suboptimal action is from a suboptimal 
    DQN model that we have trained and stopped to reach about 50% of the expert performance.
    """
    

    env = init_train_env(seed)
    random.seed(seed)
    dataset = {'observations':[], 'next_observations':[], 'actions':[], 'rewards':[],
                'terminals':[], 'timeouts':[], 'infos':[]}
    for i in range(len(train_config['topologies'])):
        observation, info = env.reset()
        done = False
        while not done:    
            rand = random.randint(0, 1)
            action = env.action_space.sample()
            last_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            dataset['observations'].append(np.array(last_observation).flatten())
            dataset['next_observations'].append(np.array(observation).flatten())
            dataset['actions'].append(np.array([action]))
            dataset['rewards'].append(reward)
            dataset['terminals'].append(terminated)
            dataset['timeouts'].append(truncated)
            dataset['infos'].append(info)

    for key in dataset:
        dataset[key] = np.array(dataset[key])
    return dataset

def store_dataset(dataset, dataset_name):
    """
    Stores dataset at filename DATA/dataset_name 

    :dataset: The dataset to be stored
    :dataset_name: The filename where the dataset should be stored. 
    """
    with open(f"DATA/{dataset_name}.pkl", 'wb') as f:
            # Serialize and save the data to the file
            pickle.dump(dataset, f)

def print_dataset(file_name):
    """
    Prints dataset and it contents as a dictionary

    :file_name: Filename of the dataset that should be printed
    """
    with open(f"DATA/{file_name}.pkl", 'rb') as f:
            # Serialize and save the data to the file
            print(f"dataset shape: {str(np.array(pickle.load(f)))}")


def convert_to_mdp_for_bc(path_dataset):
    with open(f"{path_dataset}", 'rb') as f:
            # Serialize and save the data to the file
            dataset = pickle.load(f)
            bc_dataset = MDPDataset(dataset['observations'], dataset['actions'], dataset['rewards'], dataset['terminals'],
                                action_size=3)
            
    with open(f"{path_dataset}_bc", "wb") as f:
        pickle.dump(bc_dataset, f)

if __name__ == "__main__":
    """
    Here are some examples that are commented out.
    DISCLAIMER: Make sure that the DATA folder exists.
    """

    # Generates expert dataset
    #-------------------------
    # dataset = generate_expert_dataset()
    # store_dataset(dataset, "expert_dataset")
    # print_dataset("expert_dataset")

    # Generates 10 mixed-optimal-suboptimal datasets
    #-----------------------------------------------
    # seeds = [930847, 18839, 125119, 383837, 162251]
    # for i in range(len(seeds)):
    #     dataset_name = f"suboptimal_dataset_d4rl_{i+11}"
    #     dataset = generate_suboptimal_dataset(seeds[i])
    #     store_dataset(dataset, dataset_name)
    #     print_dataset(dataset_name)

    # Converts 10 datasets with suboptimal policy to a format that can be used by BC
    #-------------------------------------------------------------------------------
    # for i in range(1, 10):
    #     convert_to_mdp_for_bc(f"DATA/suboptimal_dataset_d4rl_{i}.pkl")





