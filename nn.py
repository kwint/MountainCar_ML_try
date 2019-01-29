import argparse
import gym
import numpy as np
from itertools import count
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
# parser.add_argument('--seed', type=int, default=543, metavar='N',
#                     help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('MountainCar-v0')
# env.seed(args.seed)
# torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 200)
        # self.affine11 = nn.Linear(128, 255)
        self.affine2 = nn.Linear(200, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        # x = F.relu((self.affine11(x)))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=-1)


policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
eps = np.finfo(np.float32).eps.item()

def select_action(state, epsilon):
    if np.random.rand(1) < epsilon:
        action = np.random.randint(0,3)
    else:
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = policy(state)
        # print(probs)
        m = Categorical(probs)
        action = m.sample()
        # print(action)
        policy.saved_log_probs.append(m.log_prob(action))
        # print(action.item())
        action = action.item()
    return action


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards, device=device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def plot_durations(episode_max):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('max pos')
    plt.plot(episode_max)
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def main():
    prev_max = -2.6
    dec_max = -2.6
    dec_max_v = 0
    epsilon = 0.3
    dec_min = 0.6
    max_pos_array = []
    for i_episode in count(1):
        episode_pos = []
        episode_vel = []
        state = env.reset()
        for t in range(1000):  # Don't infinite loop while learning
            action = select_action(state, epsilon)
            state, reward, done, _ = env.step(action)
            episode_pos.append(state[0])
            episode_vel.append(state[1])
            if args.render and i_episode % 100 == 0:
                env.render()
            policy.rewards.append(reward)
            if done:
                if state[0] >= 0.5:
                    epsilon *= .99
                episode_max = max(episode_pos)
                episode_min = min(episode_pos)
                episode_max_v = abs(max(episode_vel))
                if prev_max < episode_max:
                    prev_max = episode_max
                if dec_max < episode_max:
                    dec_max = episode_max
                if dec_min > episode_min:
                    dec_min = episode_min
                if dec_max_v < episode_max_v:
                    dec_max_v = episode_max_v
                max_pos_array.append(episode_max)
                break
        policy.rewards.append((episode_max))
        # policy.rewards.append(episode_min*-1)
        # policy.rewards.append(episode_max_v*7.5)
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Ep {} \t Len {}\tpos_max {:.2f}\tpos_min {:.2f}\tv_max {:.5f}\tpos_max_all {:.5f}'.format(
                i_episode, t, dec_max, dec_min, dec_max_v, prev_max))
            dec_max = -2.6
            dec_max_v = 0
            epsilon = 0.5
            dec_min = 0.6
            plot_durations(max_pos_array)

if __name__ == '__main__':
    main()