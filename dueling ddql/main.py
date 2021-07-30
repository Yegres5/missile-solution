import gym
import multiprocessing
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from itertools import product
from dueling_ddqn_torch import Agent
from utils import PreprocessObs, reset_params, product_dict, networkParts, getData, testerGen
from visualization import drawAccuracy, printAllGraphs, drawAnimation, draw2DAnimation, Animation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
    # data = getData(nn_template=networkParts(3, 0, 1, 16), distances=(18000, 30000, 100),
    #                angles=(-60, 60, 2),
    #                maneuvers=tuple([4]), coefficients=[0])
    # names = ['Speed log', 'Overload log', 'Coord log']
    # for elems in data[:,3]:
    #     for name in names:
    #         del elems[name]
    #
    # with open("dump.npy", "wb") as f:
    #     np.save(f, data)

    # path = f"new_version_3l_aero_big_boy(4_1)/23500"
    # env, agent = testerGen(*networkParts(3, 0, 1, 16), path)
    # ini = reset_params()
    # ini["la_coord"][0] = 15000
    # ini["t_euler"][1] = np.deg2rad(45)
    # ini["maneuver"] = 4
    #
    #
    # log = play(env, agent, ini, True, 1)
    #
    # draw2DAnimation(log)
    # animator = Animation(log)
    # animator.startAnimation(100)
    #
    #
    # with open("dumpOriginal.npy", "rb") as f:
    #     data = np.load(f, allow_pickle=True)
    # #
    # # printAllGraphs(data)
    #
    # maneuvers = set([info["maneuver"] for info in data[:, 0]])
    # coefficients = set([info["coefficient"] for info in data[:, 0]])
    # limited = limitedData(data, (-60, 60), (18000, 25000))
    # indexList = getIndexList(limited, coefficients, maneuvers)
    # #
    # # printAllGraphs(limited)
    # #
    # stats = []
    # for cond, indexes in indexList.items():
    #     # m_score = np.mean(limited[indexes, 1])
    #     m_distance = np.mean([i["Distance"] for i in limited[indexes, 3]])
    #     m_accuracy = np.mean([i["Destroyed"] for i in limited[indexes, 3]])
    #     m_speed = np.mean([i["Final speed"] for i in limited[indexes, 3]])
    #     m_time = np.mean(limited[indexes, 2])
    #
    #     stats.append([cond[0], cond[1], m_distance, m_accuracy, m_speed, m_time])
    #
    # pd_stats = pd.DataFrame(stats, columns=["Вид маневра", "Коэффициент", "Дальность", "Точность",
    #                                         "Финальная скорость", "Время наведения"])
    # pd_stats.sort_values(by=["Вид маневра", "Коэффициент"])
    # pd_stats["Вид маневра"] += 1
    # pd_stats["Время наведения"] /= 10
    #
    # replace_v = {0: "Нейронная сеть"}
    # pd_stats.sort_values(by=["Вид маневра", "Коэффициент"]).replace({"Коэффициент": replace_v}).T.to_excel("output.xlsx")
    #
    # z = 0

    learning_params = list(product_dict(batch_size=[64], mem_size=[2500, 5000], gamma=[0.999], epsilon=[1],
                                        lr=[1e-5], eps_min=[0.05], decay=[300 * 700], replace=[50]))
    nn_params = list(product([3], [0], [1], [16]))

    # path = f"checkpoints/May27_15-23-45_MacBook-Pro-Evgenij.local neurons=16, main_layers=4, advantage_layers=0, value_layers=0, batch_size=128, mem_size=10000, gamma=0.999, epsilon=1, lr=1e-07, eps_min=0.05, decay=210000, replace=50/1125"
    # path = f"new_version_3l_aero_big_boy(4_1)/23500"
    # env, agent = testerGen(*networkParts(3, 0, 1, 16), path)
    #
    # writeGameToFile(env, agent, reset_params(), False, 3)

    names = [
        '__main__',
        'missile_env.envs.flying_objects',
        'missile_env.envs.missileenv_0',
        'missile_env.envs.wrapper',
        'dueling_ddqn_torch',
        'torch',
        'numpy'
    ]
    params = list(product_dict(batch_size=[64, 128, 256], mem_size=[5000], gamma=[0.999], epsilon=[1],
                               lr=[1e-5, 1e-6, 1e-7], eps_min=[0.05], decay=[300 * 700], replace=[50]))[0]

    teachArch(50000, networkParts(3, 0, 1, 16), params)

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

    # yappi.set_clock_type("wall")
    # yappi.start()

    with multiprocessing.Pool(processes=2) as pool:
        with multiprocessing.Manager() as manager:
            multiple_results = []
            for learning, (main, advantage, value, neurons) in product(learning_params, nn_params):
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
