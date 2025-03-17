import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
import random
try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
import torch

class ReplayBuffer():
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        # observation_space: spaces.Space,
        # action_space: spaces.Space,
        info: List[str],
        args: None,
        device: Union[th.device, str] = "cuda",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        # super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        self.args = args
        self.info = info
        self.device = device
        self.n_envs = n_envs
        self.R_adv_attr = False
        self._reset()

    def reset(self):
        self.returns = [0]*self.args.batch_size
        self.advantages = [0]*self.args.batch_size
        self._reset()

    def _reset(self):
        for i in self.info:
            setattr(self, i, [])
        self.buffer_size = 0
        self.wt_label = []
        # super().reset()

    def gather(self, batch_size):

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self.get(batch_size, start_idx)
            start_idx += batch_size        

    def get(self, batch_size = None, start_idx = None, shuffle = False):

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        if not shuffle:
            if start_idx is None:
                start_idx = 0
            assert start_idx<=self.buffer_size-batch_size
            idx = np.arange(start_idx, start_idx + batch_size)
            if self.R_adv_attr:
                return self._get_samples(idx),torch.stack(tuple([self.returns[i] for i in idx])), torch.stack(tuple([self.advantages[i] for i in idx]))
            else:
                return self._get_samples(idx), None, None
        else:
            idx = []
            while len(idx)<batch_size:
                idx.append(random.randint(0, self.buffer_size-1))
            if self.R_adv_attr:
                return self._get_samples(np.array(idx)), torch.stack(tuple([self.returns[i] for i in idx])), torch.stack(tuple([self.advantages[i] for i in idx]))
            else:
                return self._get_samples(idx), None, None        
    def get_sample(self, batch_size = None, start_idx = None, shuffle = False):

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        if not shuffle:
            if start_idx is None:
                start_idx = 0
            assert start_idx<=self.buffer_size-batch_size
            idx = np.arange(start_idx, start_idx + batch_size)
            return self._get_samples(idx)
        else:
            idx = []
            while len(idx)<batch_size:
                idx.append(random.randint(0, self.buffer_size-1))
            return self._get_samples(np.array(idx))    
        
    def cal_R_adv(self, len_seq = None):
        # Convert to numpy
        # last_values = last_values.clone().cpu().numpy().flatten()
        assert 'values' in self.info
        if len_seq == None:
            self.advantages = [0]*self.buffer_size
            last_gae_lam = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1
                    next_values = self.values[step]
                else:
                    next_non_terminal = 1
                    next_values = self.values[step + 1]
                delta = self.rewards[step] + self.args.gamma * next_values * next_non_terminal - self.values[step]
                last_gae_lam = delta + self.args.gamma * self.args.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
            # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
            # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
            self.returns = self.advantages + self.values
            a = 1
        else:
            self.returns = []
            self.advantages = []
            for i in range(int(self.buffer_size/len_seq)):
                self.advantages_i = [0]*len_seq
                last_gae_lam = 0
                values_list_i = self.values[i*len_seq : (i+1)*len_seq]
                rewards_list_i = self.rewards[i*len_seq : (i+1)*len_seq]
                for step in reversed(range(len_seq)):
                    if step == len_seq - 1:
                        next_non_terminal = 1
                        next_values = values_list_i[step]
                    else:
                        next_non_terminal = 1
                        next_values = values_list_i[step + 1]
                    delta = rewards_list_i[step] + self.args.gamma * next_values * next_non_terminal - values_list_i[step]
                    last_gae_lam = delta + self.args.gamma * self.args.gae_lambda * next_non_terminal * last_gae_lam
                    self.advantages_i[step] = last_gae_lam
                # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
                # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
                self.returns_i = torch.tensor(self.advantages_i) + torch.tensor(values_list_i)
                self.returns.append(self.returns_i)
                self.advantages.append(self.advantages_i)
            self.returns = [j for sub in self.returns for j in sub]
            self.advantages = [j for sub in self.advantages for j in sub]
            if self.args.cuda:
                self.returns = [tensor.cuda() for tensor in self.returns]
                self.advantages = [tensor.cuda() for tensor in self.advantages]
                # torch.tensor(self.returns, device='cuda')
            a = 1
        self.R_adv_attr = True
        



    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
        ):
        # To do: try to play with generator?
        data = []
        for i in self.info:
            if not isinstance(getattr(self, i)[0], torch.Tensor):
                j = np.array(getattr(self, i))
            else:
                j = torch.stack(tuple(getattr(self, i))).to(self.device)
                while j.dim()>2:
                    j.squeeze_(-1)
            if len(j.shape) == 1:
                if not isinstance(getattr(self, i)[0], torch.Tensor):
                    data.append(np.expand_dims(np.array(getattr(self, i))[batch_inds.tolist()], 1))
                else:
                    data.append(j[batch_inds.tolist()].unsqueeze_(1))
            else:
                data.append(j[batch_inds.tolist()])
        # data = [np.array(getattr(self, i))[:, batch_inds]for i in self.info]
        return tuple(map(self.to_torch, data))
        # return data

    def add(self, 
            data: List[Union[List, float, torch.Tensor]],
            wt_label:Union[List[bool], bool] = None,
            max_buffer_size: int = np.inf):
        while len(data) < len(self.info):
            data.append([None] * len(data[0]))
        if self.buffer_size>=max_buffer_size:
            for i in range(len(self.info)):
                j = getattr(self, self.info[i])
                j.pop(0)
        # if isinstance(data[0], List) or isinstance(data[0], np.ndarray):
        #     for i in range(len(self.info)):
        #         setattr(self, self.info[i], data[i])
        #     setattr(self, 'wt_label', wt_label)
        #     self.buffer_size = len(getattr(self, self.info[i]))
        #     assert self.buffer_size == len(wt_label)
        # else:
        for i in range(len(self.info)):
            j = getattr(self, self.info[i])
            j.append(data[i])
        self.buffer_size = len(j)
        self.wt_label.append(wt_label)

    def compute_returns(self, outlook_steps = 5, gamma = 0.99):
        R_list = [] # Return list
        for i in range(self.buffer_size - outlook_steps):
            reward_list = getattr(self, 'rewards')[i:(i+ outlook_steps)]
            R = 0
            for r in reward_list[::-1]:
                R = r + R * gamma
            R_list.append(R)
        self.add_to_buffer('returns', R_list)

    def _update(self, new_size):
        for i in self.info:
            setattr(self, i, getattr(self, i)[0:new_size])
        if 'wt_label' in self.__dict__:
            self.wt_label = self.wt_label[0:new_size]
        self.buffer_size = new_size

    def remove_redundancy(self, label):
        if not isinstance(label, List):
            label = list(label)
        for i in self.info:
            j = getattr(self, i)
            if isinstance(j, np.ndarray) and len(j.shape)>1:
                setattr(self, i, j[label, :])
            else:
                setattr(self, i, [j[q] for q in range(len(label)) if label[q]==True])
        if 'wt_label' in self.__dict__:
            self.wt_label = [self.wt_label[q] for q in range(len(self.wt_label)) if self.wt_label[q]==True]
        self.buffer_size = len(self.wt_label)

    def add_to_buffer(self, name, item):
        if not self.buffer_size == len(item):
            self._update(len(item))
        if name not in self.info:
            self.info.append(name)
        setattr(self, name, item)


    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if isinstance(array, np.ndarray):
            if copy:
                return th.tensor(array, device=self.device, dtype=torch.float64)
            return th.as_tensor(array, device=self.device, dtype=torch.float64)       
        elif isinstance(array, List):
            if not  isinstance(array[0], torch.Tensor):
                if copy:
                    return th.tensor(np.array(array), device=self.device, dtype=torch.float64)
                return th.as_tensor(np.array(array), device=self.device, dtype=torch.float64)
            else:
                torch.tensor(array)
        elif isinstance(array, torch.Tensor):
           return array.double()

# if __name__ == '__main__':
#     a = ReplayBuffer(['observation', 'actions'])
#     a.add([[np.arange(10), np.arange(10), np.arange(10)], np.arange(10)])
#     c = a.get(2, 5)
#     print(c)
#     b = 1