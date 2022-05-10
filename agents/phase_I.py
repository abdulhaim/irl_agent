import math
import random
import wandb
import torch
from copy import deepcopy
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from itertools import count

from misc.utils import get_state_gridworld
from misc.memory import ReplayMemory
from networks.discrete_gridworld import PhaseI

class PhaseIAgent():
    def __init__(self, config, env, device):
        super(PhaseIAgent, self).__init__()
        self.config = config
        self.env = env
        self.device = device
        self.memory = ReplayMemory(env.observation_shape, self.config.n_actions, size=self.config.memory_size)

        # Create Networks
        self.policy_net = PhaseI(obs=env.observation_shape,
                              n_actions=self.config.n_actions,
                              cumulants=self.config.cumulants,
                              tasks=self.config.num_tasks).to(self.device)

        self.target_net = deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)

        if self.config.load_model:
            self.policy_net.load_state_dict(torch.load("models_save/model_" + self.config.load_model_path))
            self.target_net.load_state_dict(torch.load("models_save/target_model_" + self.config.load_model_path))
            self.optimizer.load_state_dict(torch.load("models_save/optimizer_" + self.config.load_model_path))

        self.total_steps = 0

    def select_action(self, state, random_color, battery_left):
        sample = random.random()
        eps_threshold = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * \
                        math.exp(-1. * self.total_steps / self.config.eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                q, _ = self.policy_net(state.to(self.device), random_color, battery_left)
                action = q.max(1)[1]
        else:
            action = torch.tensor(random.randrange(self.config.n_actions), device=self.device, dtype=torch.long)

        log_dict = {
            "step/x_axis": self.total_steps,
            "step/action_selected": action,
            "step/epsilon": eps_threshold
        }
        wandb.log(log_dict)
        return action

    def optimize_model(self, data):
        state, action, next_state, reward, \
        next_action, color_index, next_color_index, \
        battery_level, next_battery_level, done = \
            data['state'], data['action'], data['next_state'], data['reward'], \
            data['next_action'], data['color_index'], data['next_color_index'], \
            data['battery_level'], data['next_battery_level'], data['done']

        action = action.to(torch.int64)
        reward = reward.unsqueeze(-1)
        next_action = next_action.to(torch.int64)
        color_index = color_index.long()
        next_color_index = next_color_index.long()
        battery_level = battery_level.unsqueeze(-1)
        next_battery_level = next_battery_level.unsqueeze(-1)
        done = done.unsqueeze(-1)

        assistive_q, assistive_rewards = self.policy_net(state, color_index, battery_level)
        state_action_values = torch.gather(assistive_q, 1, action)

        target_q, target_rewards = self.target_net(next_state, next_color_index, next_battery_level)
        next_state_values = torch.max(target_q, axis=-1)[0].unsqueeze(-1)
        expected_state_action_values = torch.logical_not(done) * (next_state_values * self.config.gamma) + reward
        dqn_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        computed_reward_action = torch.gather(assistive_rewards, 1, action)
        reward_loss = F.smooth_l1_loss(reward, computed_reward_action)

        phi_action = torch.gather(assistive_rewards, 1, action)
        next_psi_values = torch.gather(target_q, 1, next_action)
        expected_psi_values = torch.logical_not(done) * (next_psi_values * self.config.gamma) + phi_action.detach()

        itd_loss = F.smooth_l1_loss(state_action_values, expected_psi_values)

        log_dict = {
            "step/x_axis": self.total_steps,
            "step/dqn_loss": dqn_loss,
            "step/reward_loss": reward_loss,
            "step/itd_loss": itd_loss

        }
        wandb.log(log_dict)
        loss = dqn_loss + reward_loss + itd_loss

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def train(self, env):
        for episode in range(self.config.n_episodes):
            # Pick Color
            if self.config.multiple_tasks:
                color_index = random.randrange(self.config.num_colors)
                random_color = self.config.tasks[color_index]
            else:
                random_color = self.config.start_color
                color_index = self.config.tasks.index(random_color)

            obs = env.reset(color=random_color)
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

                self.memory.store(state.squeeze(0), action.detach(), next_state.squeeze(0), reward, next_action.detach(),
                             color_index, next_color_index,
                             battery_level, next_battery_level, done)

                state = next_state
                action = next_action
                battery_level = next_battery_level
                color_index = next_color_index

                if self.total_steps > self.config.initial_memory:
                    if self.total_steps % self.config.update_every == 0:
                        data = self.memory.sample_batch(self.config.batch_size)
                        self.optimize_model(data)

                    if self.total_steps % self.config.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

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
                torch.save(self.policy_net.state_dict(), "models/phaseI_model_" + self.config.model_name + ".pth")
                torch.save(self.optimizer.state_dict(), "models/phaseI_optimizer_" + self.config.model_name + ".pth")
                torch.save(self.target_net.state_dict(), "models/phaseI_target_model_" + self.config.model_name + ".pth")

                wandb.save("models/phaseI_model_" + self.config.model_name + ".pth")
                wandb.save("models/phaseI_optimizer_" + self.config.model_name + ".pth")
                wandb.save("models/phaseI_target_model_" + self.config.model_name + ".pth")

                if self.config.mode == "play-single" and episode == 200:
                    np.save("traj/" + "expert_" + self.config.model_name + "statey.npy", arr=self.memory.state_buf)
                    np.save("traj/" + "expert_" + self.config.model_name + "actiony.npy", arr=self.memory.action_buf)
                    np.save("traj/" + "expert_" + self.config.model_name + "next_statey.npy", arr=self.memory.next_state_buf)
                    np.save("traj/" + "expert_" + self.config.model_name + "next_actiony.npy", arr=self.memory.next_action_buf)
                    np.save("traj/" + "expert_" + self.config.model_name + "battery_levely.npy", arr=self.memory.battery_level_buf)
                    np.save("traj/" + "expert_" + self.config.model_name + "next_battery_levely.npy", arr=self.memory.next_battery_level_buf)
                    np.save("traj/" + "expert_" + self.config.model_name + "doney.npy", arr=self.memory.done_buf)


        env.close()
        return
