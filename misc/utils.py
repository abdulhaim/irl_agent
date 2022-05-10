import os
import gym
import berrygrid
import wandb
import torch
import numpy as np

def make_env(config):
    env = gym.make("MultiGrid-Color-Gather-Env-8x8-v0",
                   kwargs={'color_pick': config.start_color, 'battery_enabled': config.battery_enabled})

    env.max_episode_steps = env.max_steps
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    # initialize replay memory
    env.observation_shape = env.reset()['image'].shape
    env.observation_shape = (env.observation_shape[2], env.observation_shape[0], env.observation_shape[1])
    return env


def to_onehot(value, dim):
    """Convert batch of numbers to onehot
    Args:
        value (numpy.ndarray): Batch of numbers to convert to onehot. Shape: (batch,)
        dim (int): Dimension of onehot
    Returns:
        onehot (numpy.ndarray): Converted onehot. Shape: (batch, dim)
    """
    one_hot = torch.zeros(value.shape[0], dim)
    one_hot[torch.arange(value.shape[0]), value.long()] = 1
    return one_hot


def get_state_gridworld(true_obs):
    obs = true_obs['image']
    battery_level = true_obs['battery'][0]

    state = torch.tensor(obs)
    state = state.transpose(0, 2).transpose(1, 2)
    state = state.float().unsqueeze(0)  # swapped RGB dimension to come first
    return state, battery_level

def update_parameters(config):
    config_dict = dict()
    # Generate Red, Orange, Green Tasks
    if config.mode == "single":
        config_dict["start_color"] = "orange"
        config_dict["model_name"] = config.start_color + "_run_seed_" + str(config.seed) + config.model_name
        config_dict["multiple_tasks"] = False
        config_dict["battery_enabled"] = False
    # Play Red, Orange, Green Tasks
    elif config.mode == "play-single":
        config_dict["start_color"] = "red"
        config_dict["load_model"] = True
        config_dict["load_model_path"] = "phase_1_model_red_final_1.pth"
        config_dict["model_name"] = "play_" + config.start_color + "_run_seed_" + str(config.seed) + config.model_name
        config_dict["multiple_tasks"] = False
        config_dict["battery_enabled"] = False
        config_dict["record_episode"] = 1
        config_dict["eps_start"] = 0.02
    # Generate Multi-Task
    elif config.mode == "multitask":
        config_dict["start_color"] = "red"
        config_dict["model_name"] = "multitask_run_seed_" + str(config.seed) + config.model_name
        config_dict["load_model_path"] = "phase_1_model_multitask_final_1.pth"
        config_dict["multiple_tasks"] = True
        config_dict["battery_enabled"] = True
    # Play Multi-Task
    elif config.mode == "play-multitask":
        config_dict["start_color"] = "red"
        config_dict["load_model"] = True
        config_dict["model_name"] = "play_multitask_run_seed_" + str(config.seed) + config.model_name
        config_dict["multiple_tasks"] = True
        config_dict["battery_enabled"] = True
        config_dict["eps_start"] = 0.02
    # Generate PhaseII
    elif config.mode == "phaseII":
        config_dict["model_name"] = "red_run_seed_" + str(config.seed) + config.model_name
        config_dict["load_model"] = True
        config_dict["phaseII"] = True
        config_dict["test_sample"] = [1, 0, 0]

    config.update(config_dict, allow_val_change=True)
    return config


def generate_parameters(mode):
    # set device
    os.environ["WANDB_API_KEY"] = "83c0a7855e2613a0bba0e8bc7566b5d522729cb1"
    os.environ["WANDB_MODE"] = "online"

    # config parameters
    wandb.init()
    config = wandb.config
    config.seed = 1
    config.gamma = 0.99
    config.batch_size = 100
    config.eps_start = 1.0
    config.eps_end = 0.02
    config.eps_decay = 2000000
    config.target_update = 10000
    config.lr = 1e-4
    config.initial_memory = 2000
    config.memory_size = 10000 * config.initial_memory
    config.n_actions = 4
    config.n_episodes = 100000000
    config.record_episode = 50
    config.save_model_episode = 100
    config.cumulants = 10
    config.start_color = "red"
    config.update_every = 4
    config.phase_II_steps = 250 * 100
    config.phase_II_test_episodes = 100000
    config.mode = mode
    config.model_name = "_expert_traj"
    config.multiple_tasks = True
    config.battery_enabled = True
    config.load_model = False
    config.phaseII = False
    config.tasks = ["red", "orange", "green"]
    # config.tasks = ["red", "orange", "green", "blue", "purple", "yellow", "grey"]

    config.num_colors = len(config.tasks)
    config.num_tasks = config.num_colors
    wandb.define_metric("episode/x_axis")
    wandb.define_metric("step/x_axis")

    # set all other train/ metrics to use this step
    wandb.define_metric("episode/*", step_metric="episode/x_axis")
    wandb.define_metric("step/*", step_metric="step/x_axis")

    if not os.path.exists("models/"):
        os.makedirs("models/")

    if not os.path.exists("traj/"):
        os.makedirs("traj/")

    config = update_parameters(config)
    wandb.run.name = config.model_name

    return config
