import gymnasium as gym
import tianshou as ts
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(("Discrete SAC using Tianshou, "
        "for the meaning of some hyper-parameters, "
        "refer to the documentation of Tianshou."
    ))

    parser.add_argument('--env', default='LunarLander-v2')
    parser.add_argument('--buffer-size', default=1_000_000, type=int)
    parser.add_argument('--learning-starts', default=1000, type=int)
    parser.add_argument('--total-timesteps', default=1000000, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--learning-rate', default=3e-4, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--step-per-epoch', default=10_000, type=int)
    parser.add_argument('--step-per-collect', default=0.99, type=int)
    parser.add_argument('--update-per-step', default=0.99, type=float)
    parser.add_argument('--target-entropy-ratio', default=0.98, type=float)

    args, unknown = parser.parse_known_args()
    return args

# Network architecture, modified from cleanrl
class SoftQNetwork(nn.Module):
    def __init__(self, state_shape, actio_shape):
        super().__init__()
        self.fc1 = nn.Linear(np.array(state_shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_shape)

    def forward(self, obs, state=None, info={}):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_shape, actio_shape):
        super().__init__()
        self.fc1 = nn.Linear(np.array(state_shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_shape)

    def forward(self, obs, state=None, info={}):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logit = self.fc3(x)
        return logit, state

if __name__ == "__main__":
    kwargs = vars(parse_args())

    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(kwargs["env"]) for _ in range(1)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(kwargs["env"]) for _ in range(1)])

    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].n

    critic1, critic2 = SoftQNetwork(state_shape, action_shape), SoftQNetwork(state_shape, action_shape)
    policy = Actor(state_shape, action_shape)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=kwargs["learning_rate"])
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=kwargs["learning_rate"])
    policy_optim = torch.optim.Adam(policy.parameters(), lr=kwargs["learning_rate"])

    target_entropy = kwargs["target_entropy_ratio"] * torch.log(torch.tensor(action_shape))
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_optim = torch.optim.Adam([log_alpha], lr=kwargs["learning_rate"])

    alpha = (target_entropy, log_alpha, alpha_optim) # auto tune temperature

    policy = ts.policy.DiscreteSACPolicy(
        actor=policy,
        actor_optim=policy_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        alpha=alpha
    )

    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(kwargs["buffer_size"], 10), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    # logger
    from torch.utils.tensorboard import SummaryWriter
    from tianshou.utils import TensorboardLogger
    writer = SummaryWriter('log/dsac')
    logger = TensorboardLogger(writer)

    # training
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=kwargs["total_timesteps"]//kwargs["step_per_epoch"],
        step_per_epoch=kwargs["step_per_epoch"],
        step_per_collect=kwargs["step_per_collect"],
        update_per_step=kwargs["update_per_step"],
        episode_per_test=100, batch_size=kwargs["batch_size"], logger=logger,
        # stop_fn=lambda mean_rewards: mean_rewards >= train_envs.spec[0].reward_threshold
    )
    print(f'Finished training! Use {result["duration"]}')
