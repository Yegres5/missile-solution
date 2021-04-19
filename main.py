import gym
import numpy as np
import matplotlib.pyplot as plt
from torch_educate import PreprocessiObs, DQNAgent, evaluate, play_and_record, compute_td_loss
import torch
from replay_buffer import ReplayBuffer
import random
from IPython.display import clear_output
import utils
from tqdm import trange
import torch.nn as nn
from additional import graph
from draw_animation import drawAnimation
import sys

rocket_info = [np.array([0, 0, 0]), 900, np.deg2rad([0, 0, 0])]
target_info = [np.array([18000, 0, 0]), 200, np.deg2rad([0, 90, 0])]

ENV_NAME = "missile_env:missile-env-v0"

# m = gym.make(ENV_NAME, rocket_info=rocket_info, target_info=target_info)
# m = PreprocessiObs(m)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_env(seed, rocket_info, target_info):
    env = gym.make(ENV_NAME, rocket_info=rocket_info, target_info=target_info)  # create raw env
    if seed is not None:
        env.seed(seed)
    env = PreprocessiObs(env)
    return env


"""## Main loop
"""

def reset_params():
    return {}#{"la_coord": [np.random.uniform(-250, 250), np.random.uniform(-250, 250), np.random.uniform(-250, 250)]}

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = make_env(seed=seed, rocket_info=rocket_info, target_info=target_info)

state_shape = env.observation_space.shape
n_actions = env.action_space.n
state = env.reset(**reset_params())

agent = DQNAgent(state_shape, n_actions, epsilon=1).to(device)

target_network = DQNAgent(state_shape, n_actions).to(device)
target_network.load_state_dict(agent.state_dict())

timesteps_per_epoch = 10
batch_size = 16
total_steps = 3 * 10 ** 6
decay_steps = 1*10 ** 5/10

opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

init_epsilon = 0.8
final_epsilon = 0.01

loss_freq = 50
refresh_target_network_freq = 500
eval_freq = 1000

max_grad_norm = 50

mean_rw_history = []
td_loss_history = []
grad_norm_history = []
initial_state_v_history = []
step = 0

""" Expert play """

# exp_replay = ReplayBuffer(5*10 ** 4)
# for i in range(3000):
#     """ Ram consumable maybe check for available RAM """
#     play_and_record(initial_state=reset_params(),
#                     agent=agent, env=env, exp_replay=exp_replay, n_steps= 200, expert=True, prob_exp_random=0)
#     if len(exp_replay) == 5*10 ** 4:
#         break
#
# debug = False
# loss_log = []
# rew_log = []
#
# for i in range(100000000):
#     # train
#     play_and_record(initial_state=reset_params(),
#                     agent=agent, env=env, exp_replay=exp_replay, n_steps=500, expert=True, prob_exp_random=0)
#
#     s_, a_, r_, next_s_, done_ = exp_replay.sample(2000)
#
#     # for i in range(s_.shape[0]):
#     #     agent.update(s_[i], a_[i], r_[i], next_s_[i])
#
#     loss = compute_td_loss(s_, a_, r_, next_s_, done_, agent, target_network)
#
#     loss.backward()
#     grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
#     opt.step()
#     opt.zero_grad()
#
#     stop = False
#
#     if stop:
#         break
#
#     if i % 15 == 0:
#         # Load agent weights into target_network
#         loss_log.append(loss)
#         target_network.load_state_dict(agent.state_dict())
#         if debug == True:
#             rew_log.append(evaluate(
#                 make_env(seed=step, rocket_info=rocket_info, target_info=target_info),
#                 agent, n_games=5, greedy=True, **reset_params()))
#
#         print(loss_log[-1])#, rew_log[-1])


    # if np.floor(i / 30) == 20:
    #     break

exp_replay = ReplayBuffer(10 ** 4)

for i in range(1000):
    """ Ram consumable maybe check for available RAM """
    play_and_record(initial_state=reset_params(),
                    agent=agent, env=env, exp_replay=exp_replay, n_steps=10 ** 2, expert=False)
    if len(exp_replay) == 10 ** 4:
        break

state = env.reset(**reset_params())

for step in trange(step, total_steps + 1):
    if not utils.is_enough_ram():
        try:
            while True:
                pass
        except KeyboardInterrupt:
            pass

    agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, decay_steps)

    # all_rew = exp_replay.sample(len(exp_replay))[2]
    # mean = all_rew[all_rew != 0].mean()
    _, state = play_and_record(reset_params(), agent, env, exp_replay, timesteps_per_epoch)

    # train
    s_, a_, r_, next_s_, done_ = exp_replay.sample(200)
    # for i in range(s_.shape[0]):
    #     agent.update(s_[i], a_[i], r_[i], next_s_[i])

    loss = compute_td_loss(s_, a_, r_, next_s_, done_, agent, target_network, gamma=0.95)

    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    opt.step()
    opt.zero_grad()

    if step % loss_freq == 0:
        td_loss_history.append(loss.data.cpu().item())
        grad_norm_history.append(grad_norm)

    if step % refresh_target_network_freq == 0:
        # Load agent weights into target_network
        target_network.load_state_dict(agent.state_dict())
    if step % eval_freq == 0:
        mean_rw_history.append(evaluate(
            make_env(seed=step, rocket_info=rocket_info, target_info=target_info),
            agent, n_games=5, greedy=True, **reset_params())
        )
        initial_state_q_values = agent.get_qvalues(
            [make_env(seed=step, rocket_info=rocket_info, target_info=target_info).reset()]
        )
        initial_state_v_history.append(np.max(initial_state_q_values))

    # deb = False
    draw = False
    if step % eval_freq == 0:
    # if deb == True:
        mean_rw_history.append(evaluate(
            make_env(seed=step, rocket_info=rocket_info, target_info=target_info),
            agent, n_games=3, greedy=True, **reset_params())
        )
        initial_state_q_values = agent.get_qvalues(
            [make_env(seed=step, rocket_info=rocket_info, target_info=target_info).reset()]
        )
        initial_state_v_history.append(np.max(initial_state_q_values))

        clear_output(True)
        print("buffer size = %i, epsilon = %.5f" %
              (len(exp_replay), agent.epsilon))

        print("Last reward = ", mean_rw_history[-1])

        if draw:
            plt.figure(figsize=[16, 9])

            plt.subplot(2, 2, 1)
            plt.title("Mean reward per life")
            plt.plot(mean_rw_history)
            plt.grid()

            assert not np.isnan(td_loss_history[-1])
            plt.subplot(2, 2, 2)
            plt.title("TD loss history (smoothened)")
            plt.plot(utils.smoothen(td_loss_history))
            plt.grid()

            plt.subplot(2, 2, 3)
            plt.title("Initial state V")
            plt.plot(initial_state_v_history)
            plt.grid()

            plt.subplot(2, 2, 4)
            plt.title("Grad norm history (smoothened)")
            plt.plot(utils.smoothen(grad_norm_history))
            plt.grid()

            plt.show(block=True)

reward = 0

log = np.array(env.reset())
true_overload = np.empty(2)

for _ in range(2000):
    env.wrap.rocket.grav_compensate()
    overload = env.wrap.rocket.proportionalCoefficients(k_z=2, k_y=2)
    if true_overload.shape[0] == 0:
        true_overload = np.hstack((true_overload, overload))
    else:
        true_overload = np.vstack((true_overload, overload))

    possible = env.wrap.findClosestFromLegal(overload)
    action_num = env.wrap.overloadsToNumber([possible])[0]

    print("True = ", overload, "Possible = ", possible)
    s = env.observation(env.get_obs)
    s = torch.tensor(s, device=device, dtype=torch.float)
    action_num = torch.argmax(agent(s))

    ob, r, done, info = env.step(action_num)

    reward += r

    log = np.vstack((log, ob))
    print("Distance to target = ", env.wrap.distance_to_target)
    if done:
        break

print("Reward = ", np.round(reward, 4))

rocket_log = log[:, :15]
la_log = log[:, 15:]

graph(rocket_log=rocket_log, la_log=la_log, true_overload=true_overload)

drawAnimation()

sys.exit()
