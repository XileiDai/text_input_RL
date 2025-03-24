from typing import Any, Dict, List, Optional, Type, Union, Tuple

import torch as th
from gymnasium import spaces
from torch import nn
from transformers import GPT2Tokenizer, GPT2Model
import torch

# from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.policies import BasePolicy, BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from rl.dqn.dqn_para import Args


class q_net(nn.Module):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: Union[List[int], nn.Module] = None,
        # features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        args: Args = None,
    ):
        super().__init__()
        # super().__init__(
        #     observation_space,
        #     action_space,
        #     features_extractor=features_extractor,
        #     normalize_images=normalize_images,
        # )

        # ## remove features_extractor, replaced by LLM
        # if isinstance(features_extractor, List):
        #     features_dim = features_extractor[-1]
        #     last_layer_dim = observation_space.shape[0]
        #     fe = []
        #     for current_layer_dim in features_extractor:
        #         fe.append(nn.Linear(last_layer_dim, current_layer_dim))
        #         # if need add activation layers
        #         last_layer_dim = current_layer_dim
        #     self.fe_net = nn.Sequential(*fe)      
   
        # elif isinstance(features_extractor, nn.Module):
        #     features_dim = features_extractor[-1].out_features
        #     self.fe_net = features_extractor
        # else:
        #     features_dim = observation_space.shape[0]
        #     self.fe_net = None

        

        # Update the feature extractor to LLM model
        self.tokenizer = GPT2Tokenizer.from_pretrained('local_models/gpt2')
        self.llm_model = GPT2Model.from_pretrained('local_models/gpt2')  
        self.llm_model.to('cuda')

        for param in self.llm_model.parameters():
            # determine whether to finetune LLM
            param.requires_grad = False            

        action_dim = action_space.n
        self.share_features_extractor = share_features_extractor
        # self.n_critics = n_critics
        # self.q_networks: List[nn.Module] = []
        # for idx in range(n_critics):
        
        q_net_list = create_mlp(self.llm_model.config.n_embd, action_dim, net_arch, activation_fn) # update net_arch for the ANN, can be null list: []
        self.q_network = nn.Sequential(*q_net_list)
        self.q_network.float()
        # self.add_module(f"qf{idx}", q_net)
        # self.q_networks.append(q_net)
    
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

    def forward(self, obs: th.Tensor, state_list = None) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor

        # dxl: remove feature extractor
        # self.state_list = state_list
        inputs_token = self.generate_token(obs, state_list=state_list)
        llm_feature = self.llm_model(**inputs_token)
        last_hidden_states = llm_feature[0]  # The last hidden-state is the first element of the output tuple
        feature = last_hidden_states[:,-1,:]
        # if self.fe_net is not None:
        #     with th.set_grad_enabled(not self.share_features_extractor):
        #         feature = self.fe_net(obs.to(th.float32))
        # else:
        #     feature = obs.to(th.float32)

        qvalue_input = feature.to(th.float32)
        return self.q_network(qvalue_input)

    # def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
    #     """
    #     Only predict the Q-value using the first network.
    #     This allows to reduce computation when all the estimates are not needed
    #     (e.g. when updating the policy in TD3).
    #     """

    #     # dxl: remove feature extractor
    #     if self.fe_net is not None:
    #         with th.no_grad():
    #             feature = self.fe_net(obs.to(th.float32))
    #     else:
    #         feature = obs.to(th.float32)
    #     return self.q_networks[0](th.cat([feature, actions.to(th.float32)], dim=1))
    
    def set_training_mode(self, mode):
        self.train(mode)    


class Agent(nn.Module):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """


    q_network: q_net
    q_network_target: q_net

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Sigmoid,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = {},
        n_critics: int = 1,
        share_features_extractor: bool = True,
        args = None,
    ):
        super().__init__()
        # super().__init__(
        #     observation_space,
        #     action_space,
        #     features_extractor_class,
        #     features_extractor_kwargs,
        #     optimizer_class=optimizer_class,
        #     optimizer_kwargs=optimizer_kwargs,
        #     squash_output=True,
        #     normalize_images=normalize_images,
        # )

        # Default network architecture, from the original paper
        self.args = args
        self.optimizer_kwargs = optimizer_kwargs
        if net_arch is None:
            net_arch = [32, 8]
            net_arch = []

        _, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": observation_space,
            "action_space": action_space,
            # "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "features_extractor": [16],
            "args": self.args,
        }
        # self.actor_kwargs = self.net_args.copy()
        self.q_net_kwargs = self.net_args.copy()
        self.q_net_kwargs.update(
            {
                # "n_critics": n_critics,
                "net_arch": critic_arch,
                # "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)
        a = 1

    def _build(self, lr_schedule: Schedule) -> None:

        self.q_network = q_net(**self.q_net_kwargs)
        self.q_network_target = q_net(**self.q_net_kwargs)
        self.init_weight(self.q_network)
        self.q_network_target.load_state_dict(self.q_network.state_dict())
        self.q_network.optimizer = self.args.optimizer_class(
            self.q_network.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        # self.actor_target.set_training_mode(False)
        self.q_network_target.set_training_mode(False)
        # self.critic.float()
        # self.critic_target.float()

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                # n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def forward(self, observation: PyTorchObs, deterministic: bool = False, state_list = None) -> th.Tensor:
        # self.state_list = state_list
        return th.argmax(self.q_network(observation, state_list = state_list))

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """

        self.q_network.set_training_mode(mode)
        self.training = mode

    def init_weight(self, network):
        for m in network.q_network.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal(m.weight, mean=0, std = 0.01)
                nn.init.xavier_uniform_(m.weight)
                # nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)            