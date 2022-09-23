import gym
import multiprocessing
import os

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from itertools import product
from dueling_ddqn_torch import Agent
from utils import PreprocessObs, reset_params, product_dict, networkParts

import numpy as np


def play(env, agent, ini, expert=False, exp_coeff=5):
    score = 0
    steps = 0
    done = False
    observation = env.reset(**ini)
    overload_log = []
    speed_log = []
    coord_log = [observation[[0, 1, 2, 17, 18, 19]] * 1000]

    while not done:
        steps += 1
        if expert:
            env.wrap.rocket.grav_compensate()
            overload = env.wrap.rocket.proportionalCoefficients(k_z=exp_coeff, k_y=exp_coeff)

            possible = env.wrap.findClosestFromLegal(overload)
            action = env.wrap.overloadsToNumber([possible])[0]
        else:
            action = agent.choose_action(observation)

        observation_, reward, done, info = env.step(action)
        score += reward
        observation = observation_

        overload_log.append(env.wrap.numberToOverloads(action)[0])
        speed_log.append(env.wrap.rocketSpeed)
        coord_log.append(observation[[0, 1, 2, 17, 18, 19]] * 1000)

    info.update({"Speed log": np.array(speed_log),
                 "Overload log": np.array(overload_log),
                 "Coord log": np.array(coord_log)})

    return score, steps, info


def writeGameToFile(env, agent, init, expert=False, exp_coef=5):
    score = 0

    observations = []
    done = False
    observation = env.reset(**init)
    observations.append(observation)

    while not done:
        action = agent.choose_action(observation)

        if expert:
            env.wrap.rocket.grav_compensate()
            overload = env.wrap.rocket.proportionalCoefficients(k_z=exp_coef, k_y=exp_coef)

            possible = env.wrap.findClosestFromLegal(overload)
            action = env.wrap.overloadsToNumber([possible])[0]

        observation_, reward, done, info = env.step(action)
        print(info)
        score += reward

        observation = observation_
        observations.append(observation)

    print(score)
    observations = np.array(observations)
    rocket_coor = observations[:, 0:3] * 1000
    la_coor = observations[:, 17:20] * 1000

    np.savetxt("0.csv", rocket_coor[:, [2, 0, 1]], delimiter=",")
    np.savetxt("1.csv", la_coor[:, [2, 0, 1]], delimiter=",")
    return score


def teachArch(num_games, network_structure, params):
    network_structure = network_structure[:]
    params = dict(params)
    main_stream, advantage_stream, value_stream = network_structure
    batch_size, mem_size, gamma, epsilon, lr, eps_min, decay, replace = \
        params["batch_size"], params["mem_size"], params["gamma"], params["epsilon"], \
        params["lr"], params["eps_min"], params["decay"], params["replace"]

    print("Creating summary writer")
    tb = SummaryWriter(comment=f' neurons={main_stream[0].out_features}, '
                               f'main_layers={len(main_stream)}, '
                               f'advantage_layers={len(advantage_stream)}, '
                               f'value_layers={len(value_stream)}, '
                               f'batch_size={batch_size}, '
                               f'mem_size={mem_size}, '
                               f'gamma={gamma}, '
                               f'epsilon={epsilon}, '
                               f'lr={lr}, '
                               f'eps_min={eps_min}, '
                               f'decay={decay}, '
                               f'replace={replace}')

    env = gym.make('missile_env:missile-env-v0')

    env.seed(seed=42)
    env = PreprocessObs(env)

    state_shape = env.observation_space.shape[1]
    n_actions = env.action_space.n

    save_dir = f"checkpoints/{tb.get_logdir()[5:]}"
    full_path = os.path.join(os.getcwd(), save_dir)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr,
                  input_dims=[state_shape], n_actions=n_actions, mem_size=mem_size, eps_min=eps_min,
                  batch_size=batch_size, eps_dec=1 / decay, replace=replace,
                  chkpt_dir=save_dir,
                  main_stream=main_stream, advantage_stream=advantage_stream, value_stream=value_stream)

    scores = []
    eps_history = []
    loss_history = []

    print("Filling memory")
    while agent.memory.mem_cntr < mem_size:
        done = False
        observation = env.reset(**reset_params())
        score = 0

        while not done:
            action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action,
                                   reward, observation_, bool(done))

            observation = observation_

    for i in tqdm(range(num_games)):
        done = False
        observation = env.reset(**reset_params())
        score = 0

        while not done:
            action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            loss_history.append(agent.learn() * batch_size)

            observation = observation_

        scores.append(score)
        if i % 50 == 0:
            avg_score = np.mean(scores[max(0, i - 100):(i + 1)])
            tb.add_scalar('AVG score', avg_score, i)
            if not (loss_history[-1] is None):
                tb.add_scalar('Loss', loss_history[-1], i)
            tb.add_scalar('Scores', score, i)
            tb.add_scalar('Eps', agent.epsilon, i)

        if i > 0 and i % 125 == 0:
            agent.save_models(f"{save_dir}/{i}")

        eps_history.append(agent.epsilon)

    tb.close()


def getIndexList(data, coefficients, maneuvers):
    res = {}
    for coefficient, maneuver in product(coefficients, maneuvers):
        c_index = np.array([i["coefficient"] == coefficient for i in data[:, 0]])
        m_index = np.array([i["maneuver"] == maneuver for i in data[:, 0]])
        res[(maneuver, coefficient)] = c_index & m_index

    return res


def limitedData(data, euler_limits, coord_limits):
    angle_indexes = np.array(
        [np.deg2rad(euler_limits[0]) <= i["t_euler"][1] <= np.deg2rad(euler_limits[1]) for i in data[:, 0]])
    coord_indexes = np.array([coord_limits[0] <= i["la_coord"][0] <= coord_limits[1] for i in data[:, 0]])
    return data[angle_indexes & coord_indexes]


if __name__ == '__main__':
    learning_params = list(product_dict(batch_size=[64], mem_size=[2500, 5000], gamma=[0.999], epsilon=[1],
                                        lr=[1e-5], eps_min=[0.05], decay=[300 * 700], replace=[50]))
    nn_params = list(product([3], [0], [1], [16]))

    with multiprocessing.Pool(processes=2) as pool:
        with multiprocessing.Manager() as manager:
            multiple_results = []
            for learning, (main, advantage, value, neurons) in product(learning_params, nn_params):
                multiple_results.append(
                    pool.apply_async(teachArch, args=(50000, networkParts(main, advantage, value, neurons), learning))
                )
            [async_res.wait() for async_res in multiple_results]
