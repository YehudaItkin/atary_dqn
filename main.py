import argparse
import pickle
import time

from gym.core import ObservationWrapper
from gym.spaces import Box

from skimage.transform import resize
from skimage.color import rgb2gray
from torch.autograd import Variable

from dqn.dqn1 import DQNAgent
from utils.framebuffer import FrameBuffer
import gym
import matplotlib.pyplot as plt
import numpy as np
from utils.replay_buffer import ReplayBuffer

import torch
from tqdm import trange
from pandas import DataFrame

moving_average = lambda x, **kw: DataFrame({'x': np.asarray(x)}).x.ewm(**kw).mean().values

mean_rw_history = []
td_loss_history = []


class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def _observation(self, img):
        """what happens to each observation"""
        img = rgb2gray(img)
        img = resize(img, self.img_size[1:])[None]

        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size
        #     (use imresize imported above or any library you want,
        #      e.g. opencv, skimage, PIL, keras)
        #  * cast image to grayscale
        #  * convert image pixels to (0,1) range, float32 type

        # img /= 256.0
        return img


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        i = 0
        while True:
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            i += 1
            if done:
                break
            if i > t_max:
                print('i > t_max something wrong')
                break

        rewards.append(reward)
    return np.mean(rewards)


def make_env():
    env = gym.make("BreakoutDeterministic-v4")
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env


def play_and_record(agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time
    """
    # initial state
    obs = env.reset()
    rewards = []
    i = 0
    while True:
        i += 1
        qvalues = agent.get_qvalues([obs])
        action = agent.sample_actions(qvalues)[0]
        next_obs, r, done, _ = env.step(action)
        exp_replay.add(obs, action, r, next_obs, done)
        rewards.append(r)
        if done and i <= n_steps:
            obs = env.reset()
        elif done and i > n_steps:
            env.reset()
            break
        elif not done:
            obs = next_obs
        else:
            assert 'You sholdnt be here'

    return np.mean(rewards)


def compute_td_loss(agent, target_network, states, actions, rewards, next_states, is_done, gamma=0.99,
                    check_shapes=False):
    """ Compute td loss using torch operations only. Use the formula above. """
    device = agent.device
    states = Variable(torch.FloatTensor(states).to(device))  # shape: [batch_size, c, h, w]
    actions = Variable(torch.LongTensor(actions).to(device))  # shape: [batch_size]
    rewards = Variable(torch.FloatTensor(rewards).to(device))  # shape: [batch_size]
    next_states = Variable(torch.FloatTensor(next_states).to(device))  # shape: [batch_size, c, h, w]
    is_done = Variable(torch.FloatTensor(is_done.astype('float32')).to(device))  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    # compute V*(next_states) using predicted next q-values
    next_state_values = torch.max(predicted_next_qvalues, dim=1)[0]

    assert next_state_values.dim() == 1 and next_state_values.shape[0] == states.shape[
        0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * next_state_values

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim() == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim() == 1, "there's something wrong with target q-values, they must be a vector"

    return loss


def main(args):
    env = make_env()
    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape
    agent = DQNAgent(state_dim, n_actions, epsilon=0.5)
    target_network = DQNAgent(state_dim, n_actions)

    print('Initial fill for experience replay')
    exp_replay = ReplayBuffer(10 ** 5)
    play_and_record(agent, env, exp_replay, n_steps=10000)
    print('Done')
    opt = torch.optim.Adam(agent.parameters())
    for i in trange(10 ** 5):

        # play
        play_and_record(agent, env, exp_replay, args.play)

        # train
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(args.train)

        loss = compute_td_loss(agent,
                               target_network,
                               obs_batch,
                               act_batch,
                               reward_batch,
                               next_obs_batch,
                               is_done_batch,
                               gamma=0.99,
                               check_shapes=True)
        loss.backward()

        opt.step()

        opt.zero_grad()

        td_loss_history.append(loss.data.cpu().numpy())

        # adjust agent parameters
        if i % 500 == 0:
            agent.epsilon = max(agent.epsilon * 0.99, 0.01)
            mean_rw_history.append(evaluate(make_env(), agent, n_games=3))

        if i % 500 == 0:
            t = time.time()
            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())
            torch.save(agent.state_dict(), 'model_{}_{}.weights'.format(args.name, t))

            eps = agent.epsilon
            agent.epsilon = 0
            env_monitor = gym.wrappers.Monitor(make_env(), directory="videos_{}".format(t), force=True)
            evaluate(env_monitor, agent, n_games=1)
            env_monitor.close()
            agent.epsilon = eps

        if i % 100 == 0:
            with open('{}_mean_reward.pickle'.format(args.name), 'wb') as f:
                pickle.dump(mean_rw_history, f)

            with open('{}_td_loss.pickle'.format(args.name), 'wb') as f:
                pickle.dump(td_loss_history, f)

                print("iteration = %i, buffer size = %i, epsilon = %.5f" % (i,
                                                                            len(exp_replay),
                                                                            agent.epsilon))
            # plt.figure(figsize=[12, 4])
            # plt.subplot(1, 2, 1)
            # plt.title("mean reward per game")
            # plt.plot(mean_rw_history)
            # plt.grid()
            #
            # assert not np.isnan(td_loss_history[-1])
            # plt.subplot(1, 2, 2)
            # plt.title("TD loss history (moving average)")
            # plt.plot(moving_average(np.array(td_loss_history), span=100, min_periods=100))
            # plt.grid()
            # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN_Parser')
    parser.add_argument('--name', default='agent', type=str)
    parser.add_argument('--play', type=int, default=100)
    parser.add_argument('--train', type=int, default=256)
    args, unknwown = parser.parse_known_args()
    main(args)
