import argparse
import os
import dill
import pickle
from dataclasses import asdict, dataclass
import d3rlpy
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper


@dataclass
class TrainConfig:
    num_epochs: int = 30000 // 10  # Total number of epochs
    full_eval_every: int = 200
    eval_seed: int = 78509
    device: str = "cuda"
    results_path: str = "bc_suboptimal"
    experiment_seeds = [274199, 594953, 603901, 52289, 78509]#, 233327, 565069, 719413, 129607, 802913]
    # experiment_seeds = [274199]#, 594953, 603901, 52289, 78509]
    

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('four_room/configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

with open('four_room/configs/fourrooms_test_0_config.pl', 'rb') as file:
    test_config_0 = dill.load(file)

with open('four_room/configs/fourrooms_test_100_config.pl', 'rb') as file:
    test_config_100 = dill.load(file)

# Load the test set
test_env_0 = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                   agent_pos=test_config_0['agent positions'],
                                   goal_pos=test_config_0['goal positions'],
                                   doors_pos=test_config_0['topologies'],
                                   agent_dir=test_config_0['agent directions'],
                                   render_mode="rgb_array"))
test_env_100 = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                   agent_pos=test_config_100['agent positions'],
                                   goal_pos=test_config_100['goal positions'],
                                   doors_pos=test_config_100['topologies'],
                                   agent_dir=test_config_100['agent directions'],
                                   render_mode="rgb_array"))

results_test_100 = []
results_test_0 = []

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gpu", type=int)
args = parser.parse_args()

d3rlpy.seed(args.seed)

def eval(env, algo, n_episodes, seed, device):
    episode_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        state = state.flatten() 
        done = False
        episode_reward = 0.0
        while not done:
            action = algo.predict(
                np.array([state]), 
                # device=f"cuda:{args.gpu}"
                )[0]
            state, reward, terminated, truncated, info = env.step(action)
            state = state.flatten()
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    return np.array(episode_rewards)

def train(config: TrainConfig):
    """
    Trains BC

    :config: train config
    """
    global results_test_100, results_test_0
    # Start new experiment
    results_test_100.clear() 
    results_test_0.clear()

    bc = d3rlpy.algos.DiscreteBCConfig().create(device="cuda:0")

    for epoch in range(config.num_epochs):
        # Train for one epoch
        bc.fit(
            dataset,
            n_steps=10,
            n_steps_per_epoch=1,
            # verbose=False
        )
        
        # Evaluate every 200 epochs
        if epoch % config.full_eval_every == 0 or epoch == config.num_epochs - 1:
            # Reset environments
            test_env_100.reset(seed=config.eval_seed)
            test_env_0.reset(seed=config.eval_seed)

            eval_returns_100 = eval(
                env=test_env_100,
                algo=bc,
                n_episodes=len(test_config_100['topologies']),
                seed=config.eval_seed,
                device=config.device,
            )
            results_test_100.append(eval_returns_100)
            
            eval_returns_0 = eval(
                env=test_env_0,
                algo=bc,
                n_episodes=len(test_config_0['topologies']),
                seed=config.eval_seed,
                device=config.device,
            )
            results_test_0.append(eval_returns_0)
            

def load_dataset(file_path):
    with open(file_path, "rb") as f:
        print(f"file_path: {file_path}")
        dataset = pickle.load(f)
        
    print(str(dataset))
    print(f"dataset shape after loading: {str(np.array(dataset).shape)}")
    return dataset

def run_experiments(config: TrainConfig):
    path_100 = "eval_rewards_bc/" + config.results_path + "_100"
    path_0 = "eval_rewards_bc/" + config.results_path + "_0"
    if os.path.exists(path_100):
        with open(path_100, "rb") as f:
            all_rewards_100 = pickle.load(f)
    else:
        all_rewards_100 = []

    if os.path.exists(path_0):
        with open(path_0, "rb") as f:
            all_rewards_0 = pickle.load(f)
    else:
        all_rewards_0 = []

    for i in range(len(config.experiment_seeds)):
        global dataset
        dataset = load_dataset(f"DATA/suboptimal_dataset_d4rl_{i+11}.pkl_bc")
        config.eval_seed = config.experiment_seeds[i]
        d3rlpy.seed(config.experiment_seeds[i])
        train(config) 
        all_rewards_100.append(np.array(results_test_100))
        all_rewards_0.append(np.array(results_test_0))

    with open(path_100, "wb") as f:
        pickle.dump(all_rewards_100, f)
    with open(path_0, "wb") as f:
        pickle.dump(all_rewards_0, f)
    
    return all_rewards_100, all_rewards_0


def compare_learning_rates(all_rewards1, all_rewards2, config: TrainConfig):
    # [experiments, evals, topologies] : List[np.array]
    for i in all_rewards1:
        print(str(i.shape))
    
    for i in all_rewards2:
        print(str(i.shape))
    
    all_rewards1 = np.array(all_rewards1).sum(axis=2) 
    mean_rewards1 = np.mean(all_rewards1, axis=0)
    print(str(mean_rewards1.shape))
    std_results1 = np.std(all_rewards1, axis=0)
    confidence_intervals1 = 1.96 * std_results1 / np.sqrt(len(all_rewards1))

    all_rewards2 = np.array(all_rewards2).sum(axis=2) 
    mean_rewards2 = np.mean(all_rewards2, axis=0)
    print(str(mean_rewards2.shape))
    std_results2 = np.std(all_rewards2, axis=0)
    confidence_intervals2 = 1.96 * std_results2 / np.sqrt(len(all_rewards2))

    epochs = np.arange(0, (config.num_epochs + 1) * 10, config.full_eval_every * 10)
    print(str(epochs))
    if config.num_epochs % config.full_eval_every != 0:
        epochs = np.append(epochs, config.num_epochs)
    plt.figure(figsize=(10, 6))

    # Plot for the first model
    plt.plot(epochs, mean_rewards1, label='BC Mean Reward')
    plt.fill_between(epochs, mean_rewards1 - confidence_intervals1, mean_rewards1 + confidence_intervals1, color='b', alpha=0.2, label='BC 95% Confidence Interval')

    # Plot for the second model
    plt.plot(epochs, mean_rewards2, label='WSAC-N Mean Reward')
    plt.fill_between(epochs, mean_rewards2 - confidence_intervals2, mean_rewards2 + confidence_intervals2, color='r', alpha=0.2, label='WSAC-N 95% Confidence Interval')
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.title('Mean Rewards vs. Training Steps on Unreachable Test Set')
    plt.legend()
    plt.show()

def compare_learning_rates(all_rewards_list, model_names, config):
    """
    Compare learning rates from a list of reward arrays and model names.

    Parameters:
    all_rewards_list: List of np.array
        List containing reward arrays [experiments, evals, topologies]
    model_names: List of str
        List containing names of the models corresponding to the reward arrays
    config: TrainConfig
        Configuration object with num_epochs and full_eval_every attributes
    """
    epochs = np.arange(0, (config.num_epochs + 1) * 10, config.full_eval_every * 10)
    if config.num_epochs % config.full_eval_every != 0:
        epochs = np.append(epochs, config.num_epochs)
        
    plt.figure(figsize=(10, 6), facecolor='#1f1f1f')

    ax = plt.gca()
    ax.set_facecolor('#1f1f1f')
    
    lines = []
    for idx, (all_rewards, model_name) in enumerate(zip(all_rewards_list, model_names)):
        # Compute mean rewards and confidence intervals
        all_rewards = np.array(all_rewards).sum(axis=2)
        mean_rewards = np.mean(all_rewards, axis=0)
        std_results = np.std(all_rewards, axis=0)
        confidence_intervals = 1.96 * std_results / np.sqrt(len(all_rewards))
        
        # Plotting
        line, = plt.plot(epochs, mean_rewards, label=f'{model_name} Mean Reward')
        plt.fill_between(epochs, mean_rewards - confidence_intervals, mean_rewards + confidence_intervals, alpha=0.2)
        lines.append(line)

    # Customize plot with white axis and text
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.title('Mean Rewards vs. Training Steps on Unreachable Test Set')

    # Create a simplified legend
    handles = lines + [plt.Line2D([0], [0], color='grey', alpha=0.2, linewidth=8, label='95% Confidence Interval')]
    labels = model_names + ['95% Confidence Interval']
    ax.legend(handles, labels, facecolor='#1f1f1f', edgecolor='white', labelcolor='white')
    
    plt.show()


def plot_optimal_steps(all_rewards, config: TrainConfig):
    """
    This will plot the rewards for each experiment, assuming that you stopped training to maximize generalization performance.
    """
    # [experiments, evals, topologies]: List[np.array]
    all_rewards = np.array(all_rewards).sum(axis=2)
    max_rewards_per_experiment = np.max(all_rewards, axis=1)

    # experiments = np.arange(max_rewards_per_experiment.shape[0])
    plt.figure(figsize=(10,6))
    plt.plot(max_rewards_per_experiment, label='Max reward per experiment', marker="o", linestyle="-", color="b")
    plt.xlabel('Experiment')
    plt.ylabel('Max Reward')
    plt.title('Max rewards per experiment')
    plt.legend()
    plt.show()


def read_rewards(path):
    with open(path, "rb") as f:
        rewards = pickle.load(f)
    return rewards

if __name__ == "__main__":
    config = TrainConfig()
    # run_experiments(config)

    # Below are some examples on how to plot some results
    #----------------------------------------------------
    
    # Dataset size
    #-------------
    # rewards_bc_suboptimal_standard = read_rewards("eval_rewards_bc/bc_suboptimal_0")
    # rewards_bc_0_3x = read_rewards(f"eval_rewards_bc/bc_suboptimal_3x_0")
    # rewards_bc_0_9x = read_rewards("eval_rewards_bc/bc_suboptimal_9x_0")
    # compare_learning_rates([rewards_bc_suboptimal_standard, rewards_bc_0_3x, rewards_bc_0_9x], ["Standard size", "3x Size", "9x Size"], config)

    # Dataset quality
    #----------------
    # rewards_s_0_expert = read_rewards("eval_rewards_bc/bc_expert_0")
    # rewards_s_0_mixed = read_rewards("eval_rewards_bc/bc_mixed_suboptimal_0")
    # rewards_s_0_suboptimal = read_rewards("eval_rewards_bc/bc_suboptimal_0")
    # rewards_s_0_random = read_rewards("eval_rewards_bc/bc_random_0")
    # rewards_list = [rewards_s_0_expert, rewards_s_0_mixed, rewards_s_0_suboptimal, rewards_s_0_random]
    # compare_learning_rates(rewards_list, ["Expert", "Mixed Expert-Suboptimal", "Suboptimal", "Random"], config)

    # Comparison with expert data
    #----------------------------
    # rewards_bc_0 = read_rewards("eval_rewards_bc/bc_expert_0")
    # rewards_bc_100 = read_rewards("eval_rewards_bc/bc_expert_100")
    # rewards_s_0 = read_rewards("eval_rewards/weighted_SAC_n_expert_0")
    # rewards_s_100 = read_rewards("eval_rewards/weighted_SAC_n_expert_100")
    # compare_learning_rates([rewards_bc_0, rewards_s_0], ["BC", "WSAC-N"], config)


    # Comparison with suboptimal data
    #--------------------------------
    # rewards_bc_0 = read_rewards("eval_rewards_bc/bc_suboptimal_0")
    # rewards_bc_100 = read_rewards("eval_rewards_bc/bc_suboptimal_100")
    # rewards_s_0 = read_rewards("eval_rewards/weighted_SAC_n_suboptimal_0")
    # rewards_s_100 = read_rewards("eval_rewards/weighted_SAC_n_suboptimal_100")
    # compare_learning_rates([rewards_bc_0, rewards_s_0], ["BC", "WSAC-N"], config)

    
    


