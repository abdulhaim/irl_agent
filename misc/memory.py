import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayMemory(object):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.state_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)
        self.next_state_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.next_action_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)
        self.color_index_buf = np.zeros(size, dtype=np.float32)
        self.next_color_index_buf = np.zeros(size, dtype=np.float32)
        self.battery_level_buf = np.zeros(size, dtype=np.float32)
        self.next_battery_level_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, state, action, next_state, reward, next_action, color_index, next_color_index, battery_level, next_battery_level, done):
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.next_state_buf[self.ptr] = next_state
        self.reward_buf[self.ptr] = reward
        self.next_action_buf[self.ptr] = next_action
        self.color_index_buf[self.ptr] = color_index
        self.next_color_index_buf[self.ptr] = next_color_index
        self.battery_level_buf[self.ptr] = battery_level
        self.next_battery_level_buf[self.ptr] = next_battery_level
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            state=self.state_buf[idxs],
            action=self.action_buf[idxs],
            next_state=self.next_state_buf[idxs],
            reward=self.reward_buf[idxs],
            next_action=self.next_action_buf[idxs],
            color_index=self.color_index_buf[idxs],
            next_color_index=self.next_color_index_buf[idxs],
            battery_level=self.battery_level_buf[idxs],
            next_battery_level=self.next_battery_level_buf[idxs],
            done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}

