import gym
import numpy as np
from dueling_ddqn_torch import Agent
from gym.core import ObservationWrapper
from gym.spaces import Box
import torch.nn as nn
import os
import itertools
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import yappi
from profiler import getDataAbout
import sys
import threading
import concurrent


def polarToCart(r, phi):
    """
    phi - angle in rad
    """
    return [r * np.cos(phi), r * np.sin(phi)]


def get_rand(min_v, max_v):
    return np.random.uniform(min_v, max_v)


def get_rand_ini():
    angle = get_rand(np.deg2rad(-90), np.deg2rad(90))
    coord_angle = 0  # get_rand(np.deg2rad(-45), np.deg2rad(45))
    distance = get_rand(18000, 18000)
    la_coord = polarToCart(distance, coord_angle)
    maneuver = np.random.randint(3, 4)
    return [{"r_euler": [0, get_rand(0, 0), 0],
             "t_euler": [0, -(coord_angle + angle), 0],
             "la_coord": [la_coord[0], 0, la_coord[1]],
             "maneuver": maneuver
             }]


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(ini_list=np.empty(0))
def reset_params(count=10000):
    try:
        ini = reset_params.ini_list[0]
    except IndexError:
        reset_params.ini_list = np.hstack([reset_params.ini_list, np.hstack([get_rand_ini() for _ in range(count)])])
        ini = reset_params.ini_list[0]

    reset_params.ini_list = np.delete(reset_params.ini_list, 0)
    return ini


class PreprocessiObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.angle_values = [3, 4, 5, 29, 30, 31]
        self.coordinates_values = [0, 1, 2, 26, 27, 28]
        self.overloads = [7, 8, 9, 10, 11]  # 10,11 for navigation Ny, Nz (last values?)
        self.speed = [6, 32]  # 15, 16, 17 target speed projections
        self.relative_speed = [6]
        self.distance = [22]
        self.for_overload = [23, 24]
        self.angle_to_target = [25]
        self.target_speed = [15, 16, 17]

        self.min_coor = -25000
        self.max_coor = 25000

        self.min_rad = -np.pi
        self.max_rad = np.pi

        self.min_trig = -1
        self.max_trig = 1

        self.min_speed = 200
        self.max_speed = 900

        self.state_size = (1, self.observation(self.env.get_obs).shape[0])
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.state_size)
        self.framebuffer = np.zeros(self.state_size[1], 'float32')

    def observation(self, obs):
        obs = np.array(obs)
        all_index = self.coordinates_values + self.speed + self.distance + self.angle_to_target + [self.for_overload[0]]

        minus = np.empty(len(obs))
        div = np.empty(len(obs))
        div.fill(1)

        minus[self.coordinates_values] = 0
        minus[self.speed] = self.min_speed
        minus[self.distance] = 0
        minus[self.angle_to_target] = self.min_rad
        minus[self.for_overload[0]] = 0

        div[self.coordinates_values] = 1000
        div[self.speed] = self.max_speed - self.min_speed
        div[self.distance] = 18000
        div[self.angle_to_target] = (self.max_rad - self.min_rad)

        angles = np.reshape([np.hstack([angle, np.array(self.transform_to_trigonometry(angle))])
                             for angle in obs[self.angle_values]], -1)
        rad_norm = self.max_rad - self.min_rad
        trig_norm = self.max_trig - self.min_trig
        angles[0:angles.shape[0]:3] = (angles[0:angles.shape[0]:3] - self.min_rad) / rad_norm - 0.5
        angles[1:angles.shape[0]:3] = (angles[1:angles.shape[0]:3] - self.min_trig) / trig_norm - 0.5
        angles[2:angles.shape[0]:3] = (angles[2:angles.shape[0]:3] - self.min_trig) / trig_norm - 0.5

        additional = (obs[self.relative_speed] - obs[self.target_speed[0]]) / self.max_speed

        data_prep = ((obs - minus) / div)
        data_prep[self.angle_to_target] -= 0.5

        res = np.hstack([data_prep[all_index], angles, additional]).astype("float64")

        reorder = [0, 1, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 6, 29, 8, 10, 9,
                   3, 4, 5, 20, 21, 22, 23, 24, 25, 26, 27, 28, 7]

        res = res[reorder]
        return res.astype("float32")

    def transform_to_trigonometry(self, angle):
        return [np.sin(angle), np.cos(angle)]

    def reset(self, **info):
        return self.observation(self.env.reset(**info))
        # self.framebuffer = np.concatenate()


def play(env, agent, ini, expert=False, exp_coeff=5):
    score = 0
    steps = 0
    done = False
    observation = env.reset(**ini)
    overload_log = []
    speed_log = []
    coord_log = [observation[[0, 1, 2, 17, 18, 19]] * 10000]

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
        coord_log.append(observation[[0, 1, 2, 17, 18, 19]] * 10000)

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
            # print(overload)

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
    # la_coor = observations[:, 3:6] * 1000

    np.savetxt("0.csv", rocket_coor[:, [2, 0, 1]], delimiter=",")
    np.savetxt("1.csv", la_coor[:, [2, 0, 1]], delimiter=",")
    return score


def getCombinations(distances=(15000, 25000, 10000), angles=(-60, 60, 120), coefficients=(0, 1, 3),
                    maneuvers=(0, 1)):
    statistics = []
    combinations = list(itertools.product(
        np.arange(distances[0], distances[1] + distances[2] * 0.1, distances[2]),
        np.arange(angles[0], angles[1] + angles[2] * 0.1, angles[2]),
        coefficients, maneuvers))

    return [{
        "r_euler": [0, 0, 0],
        "t_euler": [0, np.deg2rad(i[1]), 0],
        "la_coord": [i[0], 0, 0],
        "maneuver": i[3],
        "coefficient": i[2]
    } for i in combinations]


def playThreaded(env_agent, ini, progress):
    env, agent = None, None
    while not (env and agent):
        while not len(env_agent):
            pass
        try:
            env, agent = env_agent.pop()
        except IndexError:
            pass

    score = 0
    steps = 0
    done = False
    observation = env.reset(**ini)
    overload_log = []
    speed_log = []
    coord_log = [observation[[0, 1, 2, 17, 18, 19]] * 10000]
    exp_coeff = ini["coefficient"]

    while not done:
        steps += 1
        if exp_coeff:
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

    env_agent.append((env, agent))

    print(f"Done: {progress[0].value}/{progress[1]}")
    progress[0].value += 1

    return score, steps, info


def getData(load_path="new_version_3l_aero_big_boy(4_1)/23500", distances=(10000, 30000, 1000), angles=(-90, 90, 10),
            maneuvers=tuple([2, 3, 4]), coefficients=tuple([0, 3])):
    env_agent = []
    max_threads = multiprocessing.cpu_count()

    for i in range(max_threads):
        env = gym.make('missile_env:missile-env-v0')
        env.seed(seed=42)
        env = PreprocessiObs(env)

        state_shape = env.observation_space.shape[1]
        n_actions = env.action_space.n
        agent = Agent(gamma=0.999, epsilon=1, lr=1e-4,
                      input_dims=[state_shape], n_actions=n_actions, mem_size=int(0.5 * 10 ** 4), eps_min=0.05,
                      batch_size=256, eps_dec=5 * (1e-6), replace=50)
        agent.load_models(load_path)
        agent.epsilon = 0

        env_agent.append((env, agent))

    init_list = getCombinations(distances=distances, angles=angles, maneuvers=maneuvers, coefficients=coefficients)

    with multiprocessing.Pool(processes=max_threads) as pool:
        with multiprocessing.Manager() as manager:
            env_agent = manager.list(env_agent)
            counter = manager.Value('i', 1)
            multiple_results = [pool.apply_async(playThreaded, args=(env_agent, ini, (counter, len(init_list))))
                                for i, ini in enumerate(init_list)]

            results = np.array([res.get() for res in multiple_results])

    init_list = np.array(init_list)
    init_list = np.expand_dims(init_list, axis=1)
    return np.hstack([init_list, results])


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
    env = PreprocessiObs(env)

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

    for i in range(num_games):
        print(f"{i}_{multiprocessing.current_process()}")
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

        # for name, weights in agent.q_eval.named_parameters():
        #     tb.add_histogram(name, weights, i)
        #     tb.add_histogram(f'{name}.grad', weights.grad, i)

        # print('episode: ', i, 'score %.1f ' % score,
        #       ' average score %.1f' % avg_score,
        #       'epsilon %.2f' % agent.epsilon)
        if i > 0 and i % 125 == 0:
            agent.save_models(f"{save_dir}/{i}")

        eps_history.append(agent.epsilon)

    tb.close()


def networkParts(main_layers, advantage_layers, value_layers, neurons):
    main_stream = [nn.Linear(neurons, neurons) if not (i % 2) else nn.ReLU() for i in range(main_layers * 2)]
    advantage_stream = [nn.Linear(neurons, neurons) if not (i % 2) else nn.ReLU() for i in range(advantage_layers * 2)]
    value_stream = [nn.Linear(neurons, neurons) if not (i % 2) else nn.ReLU() for i in range(value_layers * 2)]

    return main_stream, advantage_stream, value_stream


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def testerGen(main_stream, advantage_stream, value_stream, path):
    env = gym.make('missile_env:missile-env-v0')

    env.seed(seed=42)
    env = PreprocessiObs(env)
    # env = OldPreprorcessor(env)

    state_shape = env.observation_space.shape[1]
    n_actions = env.action_space.n

    # save_dir = f"checkpoints/{tb.get_logdir()[5:]}"
    # save_dir = f"checkpoints/May26_21-04-34_MBP-Evgenij.fritz.box neurons=16, main_layers=4, advantage_layers=0, value_layers=0, batch_size=256, mem_size=5000, gamma=0.999, epsilon=1, lr=0.0001, eps_min=0.05, decay=210000, replace=50"
    full_path = os.path.join(os.getcwd(), path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    agent = Agent(gamma=1, epsilon=1, lr=1,
                  input_dims=[state_shape], n_actions=n_actions, mem_size=1, eps_min=1,
                  batch_size=1, eps_dec=1 / 1, replace=1,
                  chkpt_dir=path,
                  main_stream=main_stream, advantage_stream=advantage_stream, value_stream=value_stream)
    agent.load_models(path)
    agent.epsilon = 0
    return env, agent


def profile(names, postfix="", printToFile=True):
    sort_list = ['tsub', 'ttot', 'tavg']
    mode = 'run'

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is not None:
        if gettrace():
            mode = 'debug'

    stats = getDataAbout(names)

    if printToFile:
        for sort_by in sort_list:
            file_name = f"{mode}_Optimized_{sort_by}_{postfix}"
            if file_name:
                with open(f"profile/{file_name}.txt", 'w') as f:
                    f.write(str(names))
                    stats.sort(sort_by).print_all(out=f)

    return getDataAbout(names)


if __name__ == '__main__':
    learning_params = list(product_dict(batch_size=[64, 128, 256], mem_size=[5000], gamma=[0.999], epsilon=[1],
                                        lr=[1e-5, 1e-6, 1e-7], eps_min=[0.05], decay=[300 * 700], replace=[50]))

    nn_params = list(product([3], [0], [1], [16]))
    # path = f"checkpoints/May27_15-23-45_MacBook-Pro-Evgenij.local neurons=16, main_layers=4, advantage_layers=0, value_layers=0, batch_size=128, mem_size=10000, gamma=0.999, epsilon=1, lr=1e-07, eps_min=0.05, decay=210000, replace=50/1125"
    # path = f"new_version_3l_aero_big_boy(4_1)/23500"
    # env, agent = testerGen(*networkParts(3, 0, 1, 16), path)

    # writeGameToFile(env, agent, reset_params(), False, 3)

    # missile_env.envs.flying_objects.ALL_POSSIBLE_ACTIONS

    names = [
        '__main__',
        'missile_env.envs.flying_objects',
        'missile_env.envs.missileenv_0',
        'missile_env.envs.wrapper',
        'dueling_ddqn_torch',
        'torch',
        'numpy'
    ]
    # params = list(product_dict(batch_size=[64, 128, 256], mem_size=[5000], gamma=[0.999], epsilon=[1],
    #                       lr=[1e-5, 1e-6, 1e-7], eps_min=[0.05], decay=[300 * 700], replace=[50]))[0]
    #
    # teachArch(50000, networkParts(3, 0, 1, 16), params)



    # writeGameToFile(env, agent, reset_params(), False, 3)
    # writeGameToFile(env, agent, reset_params(), True, 3)
    #
    # yappi.set_clock_type("cpu")
    # yappi.start()
    # teachArch(5, networkParts(3, 0, 1, 16), learning_params[0])  # 5
    # yappi.stop()
    #
    # profiler_data = profile([], postfix="noNames", printToFile=True)
    # profile(names, postfix="withNames", printToFile=True)

    # z = 0

    # yappi.set_clock_type("wall")
    # yappi.start()


    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        with multiprocessing.Manager() as manager:
            multiple_results = []
            for learning, (main, advantage, value, neurons) in product(learning_params, nn_params):
                print("Start process")
                multiple_results.append(
                    pool.apply_async(teachArch, args=(50000, networkParts(main, advantage, value, neurons), learning))
                )
            [async_res.wait() for async_res in multiple_results]

    # yappi.stop()
    #
    # threads = yappi.get_thread_stats()
    # modules = [sys.modules.get(module_name) for module_name in names]
    #
    # for thread in threads:
    #     yappi.get_func_stats(ctx_id=thread.id,
    #                          filter_callback=lambda x: yappi.module_matches(x, modules)
    #                          ).print_all()
    #
    # z = 0
