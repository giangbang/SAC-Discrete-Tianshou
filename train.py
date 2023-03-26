import torch
import gymnasium as gym
import tianshou as ts
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='LunarLander-v2')
    parser.add_argument('--buffer-size', default=1_000_000, type=int)
    parser.add_argument('--learning-starts', default=1000, type=int)
    parser.add_argument('--total-timesteps', default=1000000, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--gradient-steps', default=1, type=int)
    parser.add_argument('--train-freq', default=1, type=int)
    parser.add_argument('--eval-freq', default=10_000, type=int)
    parser.add_argument('--learning-rate', default=3e-4, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--target-entropy-ratio', default=0.98, type=float)

    args, unknown = parser.parse_known_args()
    return args

# Network architecture, modified from cleanrl
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.n)

    def forward(self, obs, state=None, info={}):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc1 = nn.Linear(np.array(envs.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, envs.action_space.n)

    def forward(self, obs, state=None, info={}):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logit = self.fc3(x)
        return logit, state

if __name__ == "__main__":
    kwargs = vars(parse_args())

    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(kwargs["env"]) for _ in range(1)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(kwargs["env"]) for _ in range(1)])

    state_shape = train_envs.observation_space.shape or train_envs.observation_space.n
    action_shape = train_envs.action_space.shape or train_envs.action_space.n

    critic1, critic2 = SoftQNetwork(train_envs), SoftQNetwork(train_envs)
    policy = Actor(train_envs)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=kwargs["learning_rate"])
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=kwargs["learning_rate"])
    policy_optim = torch.optim.Adam(policy.parameters(), lr=kwargs["learning_rate"])

    target_entropy = kwargs["target_entropy_ratio"] * torch.log(torch.tensor(train_envs.action_space.n))
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_optim = torch.optim.Adam([log_alpha], lr=kwargs["learning_rate"])

    alpha = (target_entropy, log_alpha, alpha_optim) # auto tune temperature

    policy = ts.policy.DiscreteSACPolicy(policy, policy_optim,
        critic1, critic1_optim,
        critic2, critic2_optim,
        alpha)

    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    # logger
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    writer = SummaryWriter('log/dsac')
    logger = TensorboardLogger(writer)

    # training
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=10000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64, logger=logger,
        stop_fn=lambda mean_rewards: mean_rewards >= train_envs.spec.reward_threshold)
    print(f'Finished training! Use {result["duration"]}')
