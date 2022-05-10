import math
import random
import wandb
import torch
from copy import deepcopy
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from misc.memory import ReplayMemory
from misc.utils import get_state_gridworld
from itertools import count

from misc.utils import to_onehot
from networks.discrete_gridworld import PhaseI, PhaseII


class PhaseIIAgent():
    def __init__(self, config, env, device):
        super(PhaseIIAgent, self).__init__()
        self.config = config
        self.env = env
        self.device = device
        self.memory = ReplayMemory(env.observation_shape, self.config.n_actions, size=self.config.memory_size)
        self.memory.state_buf = np.load("traj/" + "expert_play_" + self.config.model_name + "statey.npy")
        self.memory.action_buf = np.load("traj/" + "expert_play_" + self.config.model_name + "actiony.npy")
        self.memory.next_state_buf = np.load("traj/" + "expert_play_" + self.config.model_name + "next_statey.npy")
        self.memory.next_action_buf = np.load("traj/" + "expert_play_" + self.config.model_name + "next_actiony.npy")
        self.memory.battery_level_buf = np.load("traj/" + "expert_play_" + self.config.model_name + "battery_levely.npy")
        self.memory.next_battery_level_buf = np.load("traj/" + "expert_play_" + self.config.model_name + "next_battery_levely.npy")
        self.memory.done_buf = np.load("traj/" + "expert_play_" + self.config.model_name + "doney.npy")
        self.memory.size = len(self.memory.done_buf)
        # Create Networks
        self.policy_net = PhaseII(obs=env.observation_shape,
                              n_actions=self.config.n_actions,
                              cumulants=self.config.cumulants,
                              tasks=self.config.num_tasks).to(self.device)

        self.target_net = deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

        self.load_models()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.total_steps = 0

    def load_models(self):
        self.config.load_phaseI_model_path = "_phase_1_model_multitask_final_1.pth"
        policy_state_dict = torch.load("models_phaseI/model" + self.config.load_phaseI_model_path)
        target_state_dict = torch.load("models_phaseI/target_model" + self.config.load_phaseI_model_path)
        optimizer_state_dict = torch.load("models_phaseI/optimizer" + self.config.load_phaseI_model_path)
        self.optimizer.load_state_dict(optimizer_state_dict)

        old_policy_net = PhaseI(obs=self.env.observation_shape,
                              n_actions=self.config.n_actions,
                              cumulants=self.config.cumulants,
                              tasks=self.config.num_tasks).to(self.device)
        old_policy_net.load_state_dict(policy_state_dict)

        old_target_net = PhaseI(obs=self.env.observation_shape,
                              n_actions=self.config.n_actions,
                              cumulants=self.config.cumulants,
                              tasks=self.config.num_tasks).to(self.device)
        old_target_net.load_state_dict(target_state_dict)

        self.policy_net.features = old_policy_net.features
        self.policy_net.psi = old_policy_net.psi
        self.policy_net.phi = old_policy_net.phi
        self.policy_net.w = nn.Parameter(torch.mean(old_policy_net.w, axis=0))

        self.target_net.features = old_target_net.features
        self.target_net.psi = old_target_net.psi
        self.target_net.phi = old_target_net.phi
        self.target_net.w = nn.Parameter(torch.mean(old_target_net.w, axis=0))

        self.optimizer.load_state_dict(optimizer_state_dict)

    def select_action(self, state, random_color, battery_left):
        with torch.no_grad():
            q, _ = self.policy_net(state.to(self.device), random_color, battery_left)
            action = q.max(1)[1]

        log_dict = {
            "step/x_axis": self.total_steps,
            "step/action_sself.policy_net.welected": action
        }
        wandb.log(log_dict)
        return action

    def optimize_model_phase_II(self, data):
        state, action, next_state, next_action, battery_level, next_battery_level, done = \
            data['state'], data['action'], data['next_state'], \
            data['next_action'], data['battery_level'], data['next_battery_level'], data['done']

        action = action.to(torch.int64)
        next_action = next_action.to(torch.int64)
        color_index = torch.tensor(self.config.test_sample).repeat(self.config.batch_size, 1).long().unsqueeze(-1)
        battery_level = battery_level.unsqueeze(-1)
        next_battery_level = next_battery_level.unsqueeze(-1)
        done = done.unsqueeze(-1)

        assistive_q, assistive_rewards = self.policy_net(state, color_index, battery_level)
        state_action_values = torch.gather(assistive_q, 1, action)
        phi_action = torch.gather(assistive_rewards, 1, action)
        target_q, _ = self.target_net(next_state, color_index, next_battery_level)
        next_psi_values = torch.gather(target_q, 1, next_action)
        expected_psi_values = torch.logical_not(done) * (next_psi_values * self.config.gamma) + phi_action.detach()

        itd_loss = F.smooth_l1_loss(state_action_values, expected_psi_values)
        bc_loss = self.nll_loss_fn(action, assistive_q)
        loss = bc_loss + itd_loss

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def nll_loss_fn(self, action, q_next):
        # only for human - cross entropy between (one-hot) real action of human and the softmax of q of human.
        one_hot_action = to_onehot(action, self.policy_net.n_actions)
        softmax_function = nn.Softmax(dim=-1)
        softmax_q = softmax_function(q_next)
        assert one_hot_action.shape == softmax_q.shape
        return self.cross_entropy_loss(one_hot_action, softmax_q)

    def inverse_train(self):
        for step in range(self.config.phase_II_steps):
            self.total_steps = step
            data = self.memory.sample_batch(self.config.batch_size)
            self.optimize_model_phase_II(data)
            if self.total_steps % self.config.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            print('Total steps: {}'.format(self.total_steps))

            if step % self.config.save_model_episode == 0:
                torch.save(self.policy_net.state_dict(), "models/" + self.config.model_name + ".pth")
                wandb.save("models/" + self.config.model_name + ".pth")

    def test_agent(self, env):
        color_index = 1
        self.total_steps = 0
        for episode in range(self.config.phase_II_test_episodes):
            # Pick Color
            obs = env.reset(color="red")
            state, battery_level = get_state_gridworld(obs)
            action = self.select_action(state, color_index, battery_level)
            total_reward = 0.0
            for t in count():
                self.total_steps += 1
                obs, reward, done, info = env.step(action)
                reward = reward[0]

                total_reward += reward
                next_state, next_battery_level = get_state_gridworld(obs)
                next_color_index = color_index
                next_action = self.select_action(next_state, next_color_index, next_battery_level)

                action = next_action
                color_index = next_color_index

                if done:
                    break

            if episode % self.config.record_episode == 0:
                log_dict = {
                    "episode/x_axis": episode,
                    "episode/episodic_reward_episode": total_reward,
                    "episode/length": t,
                    "step/x_axis": self.total_steps,
                    "step/episodic_reward_steps": total_reward
                }
                wandb.log(log_dict)

                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.total_steps, episode, t,
                                                                                     total_reward))

            if episode % self.config.save_model_episode == 0:
                torch.save(self.policy_net.state_dict(), "models/phaseII_model_" + self.config.model_name + ".pth")
                torch.save(self.optimizer.state_dict(), "models/phaseII_optimizer_" + self.config.model_name + ".pth")
                torch.save(self.target_net.state_dict(), "models/phaseII_target_model_" + self.config.model_name + ".pth")

                wandb.save("models/phaseII_model_" + self.config.model_name + ".pth")
                wandb.save("models/phaseII_optimizer_" + self.config.model_name + ".pth")
                wandb.save("models/phaseII_target_model_" + self.config.model_name + ".pth")

        env.close()
        return
