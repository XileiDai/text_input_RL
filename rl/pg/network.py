import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np
from stable_baselines3.common.policies import BasePolicy
import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from gymnasium.spaces import (
    Box,
    Discrete
)
from transformers import GPT2Tokenizer, GPT2Model

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from rl.util.FeatureExt_network import FEBuild_actor
from rl.util.build_network import MlpBuild_actor
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from rl.util.schedule import ConstantSchedule
import torch

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class FlattenHead(nn.Module):
    def __init__(self, nf, action_dim, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        # self.linear = nn.Linear(nf, action_dim)
        # self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        # x = self.linear(x)
        # x = self.dropout(x)
        return x

class Agent(nn.Module):
    def __init__(self, 
                observation_space: spaces.Space,
                action_space: spaces.Space,
                lr_schedule: Schedule,
                net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                Fe_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                optimizer_class: Type[th.optim.Optimizer] = th.optim.SGD,
                activation_fn: Type[nn.Module] = nn.Tanh,
                extract_features_bool: bool = True,
                share_features_extractor: bool = True,
                device: Union[str, torch.device] = 'cuda',
                optimizer_kwargs: Optional[Dict[str, Any]] = None,
                use_sde: bool = False,
                ortho_init: bool = False,
                xa_init_gain: float = 0.5,
                ):
        super(Agent, self).__init__()
        self.activation_fn = activation_fn
        self.observation_space = observation_space
        self.action_space = action_space
        self.extract_features_bool = extract_features_bool
        self.device = device
        self.ortho_init = ortho_init
        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        
        
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        if optimizer_kwargs is None:
            self.optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                self.optimizer_kwargs["eps"] = 1e-5
        self.lr_schedule = lr_schedule
        # Default network architecture, from stable-baselines
        if net_arch is None:
                net_arch = dict(pi=[32])
        self.net_arch = net_arch

        if Fe_arch is None:
                Fe_arch = [64, 32]
        # self.features_extractor = FEBuild_actor(
        #     self.observation_space.shape[0],
        #     Fe_arch = Fe_arch,
        #     # activation_fn=self.activation_fn,
        #     device=self.device,
        # )
        # self.mlp_extractor = MlpBuild_actor(
        #     self.features_extractor.latent_dim_pi,
        #     net_arch=self.net_arch,
        #     activation_fn=self.activation_fn,
        #     device=self.device,
        # )
        self.tokenizer = GPT2Tokenizer.from_pretrained('local_models/gpt2')
        self.llm_model = GPT2Model.from_pretrained('local_models/gpt2')  
        self.llm_model.to('cuda')        
        for param in self.llm_model.parameters():
            # determine whether to finetune LLM
            param.requires_grad = False                 
        self.d_ff = 32
        self.flat = FlattenHead(nf=self.d_ff*256, action_dim = 3, head_dropout = 0.1)

        if type(self.action_space) == Discrete:
            self.dist_type = 'categorical'
            self.action_network = nn.Sequential(
                                        nn.Linear(self.d_ff*256, self.action_space.n),
                                        nn.Softmax(dim =-1),
                                        ).to(self.device)            
        elif type(self.action_space) == Box:
            self.dist_type = 'normal'
            self.action_network_mu = nn.Sequential(
                                        nn.Linear(self.d_ff*256, 1),
                                        nn.Sigmoid(),
                                        ).to(self.device)       
            self.action_network_logstd = nn.Sequential(
                                        nn.Linear(self.d_ff*256, 1),
                                        nn.Sigmoid(),
                                        ).to(self.device)                           
        # self.action_dist = CategoricalDistribution(self.action_space.n)
      
        # self.xa_init_gain = xa_init_gain
        if self.dist_type == 'categorical':
            self.init_weight(self.action_network)
        if self.dist_type == 'normal':
            self.init_weight(self.action_network_mu)
            self.init_weight(self.action_network_logstd)
        # self.init_weight(self.mlp_extractor.policy_net)
        
        self._build(lr_schedule)
        self._load_pre_train(False)

    def _load_pre_train(self, bool = False):
        if bool:
            self.load_state_dict(torch.load('Archive results\\0-pg-auto-important\\PG-v1__buildinggym-PG__1__1718099220\\model.pth'))

    def generate_prompt(self, obs, state_list = None):
        with open('Prompt template/Prompt.txt', 'r') as file:
            # Read the entire content of the file
            prompt_templatte = file.read()
        # self.inter_obs_var = ['t_out', 't_in', 'occ', 'light', 'Equip']

        # Outdoor temperature: {t_out} °C;
        # Indoor temperature: {t_in} °C;
        # People density per floor area: {people_var} W/m²;
        # Lighting power per floor area: {light_var} W/m²;
        # Electrical equipment power per floor area: {equipment_var} W/m²; 
        occ = 4.30556417E-02 * state_list[2]
        light = 15.59* state_list[3]
        Equip = 8.07293281E+00* state_list[4]
        prompt_templatte = prompt_templatte.replace('{t_out}', str(round(state_list[0],2)))
        prompt_templatte = prompt_templatte.replace('{t_in}', str(round(state_list[1],2)))
        prompt_templatte = prompt_templatte.replace('{people_var}', str(round(occ,2)))
        prompt_templatte = prompt_templatte.replace('{light_var}', str(round(light,2)))
        prompt_templatte = prompt_templatte.replace('{equipment_var}', str(round(Equip,2)))
        return prompt_templatte


    def generate_token(self, obs, state_list):
        input_ids = []
        attention_mask = []
        if len(obs.shape)>1:
            for i in range(obs.shape[0]):
                prompt = self.generate_prompt(obs[i], state_list = state_list[i,:])
                inputs_token = self.tokenizer(prompt, return_tensors="pt")
                inputs_token = {key: value.to("cuda") for key, value in inputs_token.items()}
                input_ids.append(inputs_token['input_ids'])
                attention_mask.append(inputs_token['attention_mask'])
            input_ids = torch.stack(input_ids).squeeze()
            attention_mask = torch.stack(attention_mask).squeeze()
            tokens = {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            prompt = self.generate_prompt(obs, state_list=state_list)
            tokens = self.tokenizer(prompt, return_tensors="pt")
            tokens = {key: value.to("cuda") for key, value in tokens.items()}
        return tokens

    def set_training_mode(self, mode = True):
        if mode:
            self.train()
        else:
            self.eval()

    def forward(self, obs: th.Tensor, deterministic: bool = False, state_list = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        # if self.extract_features_bool:
        #     pi_features = self.features_extractor.extract_features(obs)
        # else:
        #     pi_features = obs
        # # if self.share_features_extractor:
        # #     latent_pi, latent_vf = self.features_extractor(features)
        # # else:
        # #     pi_features=vf_features = features
        # latent_pi = self.mlp_extractor(pi_features)
        inputs_token = self.generate_token(obs, state_list=state_list)
        llm_feature = self.llm_model(**inputs_token)
        last_hidden_states = llm_feature[0]  # The last hidden-state is the first element of the output tuple
        feature = last_hidden_states[:,:,:self.d_ff]
        feature = self.flat(feature)
        # Evaluate the values for the given observations
        distribution = self._get_action_dist_from_latent(feature)
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        # actions = actions.reshape((-1, *self.action_space.n)) 
        return actions, None, log_prob
    

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """


        if self.dist_type == 'categorical':
            mean_actions = self.action_network(latent_pi)
            return Categorical(mean_actions)
        if self.dist_type == 'normal':
            mu = self.action_network_mu(latent_pi)
            std = self.action_network_logstd(latent_pi)      
            return Normal(mu.squeeze(), std.squeeze()*0.3)
        # if isinstance(self.action_dist, DiagGaussianDistribution):
        #     return self.action_dist.proba_distribution(mean_actions, self.log_std)
        # elif isinstance(self.action_dist, CategoricalDistribution):
        #     # Here mean_actions are the logits before the softmax
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, MultiCategoricalDistribution):
        #     # Here mean_actions are the flattened logits
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, BernoulliDistribution):
        #     # Here mean_actions are the logits (before rounding to get the binary actions)
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        # else:
        #     raise ValueError("Invalid action distribution")

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor, state_list = None) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        # if self.extract_features_bool:
        #     pi_features = self.features_extractor.extract_features(obs)
        # else:
        #     pi_features = obs
        # # if self.share_features_extractor:
        # #     pi_features, vf_features = self.features_extractor.extract_features(obs)
        # # else:
        # #     pi_features = features[0]
        # #     vf_features = features[1]
        # latent_pi = self.mlp_extractor(pi_features)
        inputs_token = self.generate_token(obs, state_list=state_list)
        llm_feature = self.llm_model(**inputs_token)
        last_hidden_states = llm_feature[0]  # The last hidden-state is the first element of the output tuple
        feature = last_hidden_states[:,:,:self.d_ff]
        feature = self.flat(feature)        
        distribution = self._get_action_dist_from_latent(feature)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        if self.extract_features_bool:
            features = self.features_extractor.extract_features(obs)
        else:
            features = obs
        latent_pi = self.mlp_extractor(features)
        return self._get_action_dist_from_latent(latent_pi)

      
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        # latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # if isinstance(self.action_dist, DiagGaussianDistribution):
        #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        #     )
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
        #     )
        # elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
        #     self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        # else:
        #     raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        # self.action_net.to(self.policydevice)
        # self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1, device=self.policydevice)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_network: 0.1,
            }
            
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]    
        # self.optimizer =torch.optim.SGD(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]    

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=self.xa_init_gain, std=0.5)
            # torch.nn.init.zero_(m.bias)

    # define init method inside your model class
    def init_weight(self, network):
        for m in network.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.001)
                # nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)    
