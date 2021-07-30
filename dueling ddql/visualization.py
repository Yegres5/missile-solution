import pandas
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import gridspec
from itertools import product
from matplotlib import animation
from math import floor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import static_vars


def Gen_RandLine(length, dims=2):
    lineData = np.empty((dims, length))
    lineData[:, 0] = np.random.rand(dims)
    for index in range(1, length):
        step = ((np.random.rand(dims) - 0.5) * 0.1)
        lineData[:, index] = lineData[:, index - 1] + step
    return lineData


def update_lines(num, dataLines, lines, ax):
    for line, data in zip(lines, dataLines):
        delta = 5000.
        x0 = dataLines[0][0, num]
        y0 = dataLines[0][1, num]
        z0 = dataLines[0][2, num]

        ax.set_xbound([-delta + x0, delta + x0])
        ax.set_ybound([-delta + y0, delta + y0])
        ax.set_zbound([-delta + z0, delta + z0])

        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        # if np.allclose(data, dataLines[-1])
    return lines


class Animation():
    def __init__(self, data):
        # self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3)
        self.fig = plt.figure()
        # self.ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        # self.ax2 = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=2)
        # self.ax3 = plt.subplot2grid((2, 4), (1, 2), rowspan=2, colspan=2)

        AX = gridspec.GridSpec(2, 4)
        AX.update(wspace=0.5, hspace=0.5)
        self.ax1 = plt.subplot(AX[:, 0:2])
        self.ax2 = plt.subplot(AX[0, 2:])
        self.ax3 = plt.subplot(AX[1, 2:])

        self.ax2.title.set_text("Зависимость скорости от времени")
        self.ax3.title.set_text("Зависимость перегрузки от времени")
        self.ax2.set_ylabel('Скорость')
        self.ax2.set_xlabel('Время')
        self.ax3.set_ylabel('Перегрузка')
        self.ax3.set_xlabel('Время')

        self.data = data
        self.coords = data[2]["Coord log"][:, [0, 2, 3, 5]]
        self.over = data[2]["Overload log"]
        self.spd = data[2]["Speed log"]
        self.timeLine = np.arange(0.1, len(self.over) / 10 + 0.1, 0.1)
        self.min_v, self.max_v = np.min(self.coords), np.max(self.coords)
        self.min_t, self.max_t = 0, np.ceil(np.max(self.timeLine)/10)*10
        self.min_spd, self.max_spd = np.round(min(self.spd), -1), np.round(max(self.spd), -1)
        self.min_over, self.max_over = min(self.over), max(self.over)

        self.ax1.set_xlim(self.min_v, self.max_v)
        self.ax1.set_ylim(self.min_v, self.max_v)
        self.ax2.set_xlim(self.min_t - 1, self.max_t)
        self.ax2.set_ylim(self.min_spd - 10, self.max_spd + 10)
        self.ax3.set_xlim(self.min_t - 1, self.max_t + 1)
        self.ax3.set_ylim(self.min_over - 1, self.max_over + 1)

        self.ax2.yaxis.set_ticks(np.linspace(np.round(self.min_spd), np.round(self.max_spd), 5, endpoint=True))
        self.ax2.xaxis.set_ticks(np.linspace(np.round(self.min_t), np.round(self.max_t), 11, endpoint=True))
        self.ax3.yaxis.set_ticks(np.linspace(np.ceil(self.min_over), np.floor(self.max_over), 5, endpoint=True))
        self.ax3.xaxis.set_ticks(np.linspace(np.round(self.min_t), np.round(self.max_t), 11, endpoint=True))

        self.rocket = self.ax1.plot([self.coords[0, 0], self.coords[0, 1]])[0]
        self.la = self.ax1.plot([self.coords[0, 2], self.coords[0, 3]])[0]

        self.overload = self.ax3.plot([self.timeLine[0], self.over[0]])[0]

        self.speed = self.ax2.plot([self.timeLine[0], self.spd[0]])[0]
        # self.ax1.plot([-40000, 40000], [0, 0], 'k-.', linewidth=0.3)
        # self.ax1.plot([0, 0], [-40000, 40000], 'k-.', linewidth=0.3)

    def animate(self, i):
        self.rocket.set_data(self.coords[:i, 0], self.coords[:i, 1])
        self.la.set_data(self.coords[:i, 2], self.coords[:i, 3])

        self.overload.set_data(self.timeLine[:i], self.over[:i])
        self.speed.set_data(self.timeLine[:i], self.spd[:i])

        if i != 0 and i < self.coords.shape[0]:
            new_x_min = min(self.coords[i - 1, [0, 2]])
            new_x_max = max(self.coords[i - 1, [0, 2]])
            x_center = np.mean([new_x_min, new_x_max])

            diff_x = abs(self.max_v - new_x_min)

            new_y_min = min(self.coords[i - 1, [1, 3]])
            new_y_max = max(self.coords[i - 1, [1, 3]])
            diff_y = abs(self.max_v - new_x_min)
            y_center = np.mean([new_y_min, new_y_max])

            max_diff = max([diff_y, diff_x])

            # self.ax1.set_xlim(x_center - max_diff / 2 - 5000, x_center + max_diff / 2 + 5000)
            # self.ax1.set_ylim(y_center - max_diff / 2 - 5000, y_center + max_diff / 2 + 5000)

            self.ax1.set_xlim(0, 25000)
            self.ax1.set_ylim(-10000, 10000)

    def startAnimation(self, interval=1000):
        ani = animation.FuncAnimation(self.fig, self.animate, interval=3, frames=600, repeat=True)
        plt.show()


def draw2DAnimation(data):
    # fig, (ax1, ax2) = plt.subplots(2)
    coords = data[2]["Coord log"][:, [0, 2, 3, 5]] / 10
    over = data[2]["Overload log"]
    spd = data[2]["Speed log"]

    # timeLine = np.arange(0.1, len(over)/10+0.1, 0.1)
    #
    # ax1.plot(timeLine, spd)
    #
    # ax2.plot(timeLine, over)

    fig, ax = plt.figure()

    return


def drawAnimation():
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Turn off axis
    ax.set_axis_off()

    # Create axes x y z
    xspan, yspan, zspan = 3 * [np.linspace(0, 20000, 20)]
    zero = np.zeros_like(xspan)

    ax.plot3D(xspan, zero, zero, 'k-.', linewidth=0.8)
    ax.plot3D(zero, yspan, zero, 'k-.', linewidth=0.8)
    ax.plot3D(zero, zero, zspan, 'k-.', linewidth=0.8)

    ax.text(xspan.max() + 10, .5, .5, "Z", color='red', fontsize=20)
    ax.text(.5, yspan.max() + 10, .5, "X", color='red', fontsize=20)
    ax.text(.5, .5, zspan.max() + 10, "Y", color='red', fontsize=20)

    # Getting CSV data
    column_names = ['x', 'y', 'z']
    list = []
    for i in range(2):
        list.append(pandas.read_csv("%d.csv" % (i), header=None))

    # Creating arrays of data from CSV
    data = []
    for la in list:
        data.append(np.array(la.values).transpose())

    # Creating fifty line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    lines = [ax.plot(dat[2, 0:1], dat[0, 0:1], dat[1, 0:1])[0] for dat in data]

    # X-Z Y-X Z-Y

    # Setting the axes properties
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    # Setting x0 y0 z0
    delta = 10000.
    x0 = 0
    y0 = 0
    z0 = 0

    ax.set_xbound([-delta + x0, delta + x0])
    ax.set_ybound([-delta + y0, delta + y0])
    ax.set_zbound([-delta + z0, delta + z0])

    speed = 1
    indexes = np.arange(floor(data[0].shape[1] / speed)) * speed
    if indexes[-1] != data[0].shape[1] - 1:
        indexes = np.hstack([indexes, np.array([data[0].shape[1] - 1])])

    data = [d[:, indexes] for d in data]

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, data[0].shape[1], fargs=(data, lines, ax),
                                       interval=10, blit=False, repeat=False)

    plt.show(block=True)


@static_vars(ALL_NAMES=["Score nonbin", "Score bin", "Distance", "Distance scaled",
                        "Time", "Time if hit", "Speed", "Speed if hit", "PN vs NN Distance",
                        "PN vs NN Distance diff", "PN vs NN Accuracy", "Fancy", "PN vs NN Fancy"])
def drawAccuracy(stats, maneuver, coefficient, graphType="Score nonbin", save_path=None):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    indexes_maneuver = np.array([i["maneuver"] for i in stats[:, 0]]) == maneuver
    indexes = np.array([i["coefficient"] for i in stats[:, 0]]) == coefficient
    indexes_neural = np.array([i["coefficient"] for i in stats[:, 0]]) == 0

    indexes = indexes_maneuver & indexes
    indexes_neural = indexes_maneuver & indexes_neural

    distances = [i["la_coord"][0] for i in stats[indexes, 0]]
    angles = [np.round(np.rad2deg(i["t_euler"][1])) for i in stats[indexes, 0]]

    hit_or_miss = stats[indexes, 1].astype("float64")
    hit_or_miss[hit_or_miss < 0] = 0

    cmap = mpl.cm.bwr
    min_score, max_score = None, None

    if graphType == "Score nonbin":  # nope
        scores = stats[indexes, 1].astype("float64")

    if graphType == "Score bin":
        scores = stats[indexes, 1].astype("float64")
        scores[scores < 0] = 0

    elif graphType == "Distance":  # nope
        scores = [i["Distance"] for i in stats[indexes, 3]]
        min_score, max_score = 100, 1000

    elif graphType == "Distance scaled":
        cmap = mpl.cm.coolwarm
        scores = np.array([i["Distance"] for i in stats[indexes, 3]], dtype=np.float)
        scores[scores < 100] = 100
        j, c, i, q = -1 / 2, 1000, 1500, -2
        scores = j * (np.tanh(1 / c * (scores - i)) + (q + 1))
        min_score, max_score = 0, 1

    elif graphType == "Time":
        scores = stats[indexes, 2].astype("float64")

    elif graphType == "Time if hit":
        scores = stats[indexes, 2].astype("float64")
        scores[hit_or_miss == 0] = 600

    elif graphType == "Speed":
        scores = np.array([i["Final speed"] for i in stats[indexes, 3]])
        min_score, max_score = 200, 900

    elif graphType == "Speed if hit":
        scores = np.array([i["Final speed"] for i in stats[indexes, 3]])
        scores = scores * hit_or_miss
        min_score, max_score = 200, 900

    elif graphType == "PN vs NN Distance":  # nope
        pn_dist = [i["Distance"] for i in stats[indexes, 3]]
        nn_dist = [i["Distance"] for i in stats[indexes_neural, 3]]
        if not len(nn_dist):
            return
        scores = [1 if nn_d < pn_d else 0 for pn_d, nn_d in zip(pn_dist, nn_dist)]

    elif graphType == "PN vs NN Distance diff":
        pn_dist = np.array([i["Distance"] for i in stats[indexes, 3]])
        nn_dist = np.array([i["Distance"] for i in stats[indexes_neural, 3]])
        if not len(nn_dist):
            return
        scores = nn_dist - pn_dist

    elif graphType == "PN vs NN Accuracy":
        min_score, max_score = 0, 100
        cmap = mpl.cm.brg

        pn_score = stats[indexes, 1].astype("float64")
        nn_score = stats[indexes_neural, 1].astype("float64")
        if not len(nn_score):
            return

        scores = []
        for pn_d, nn_d in zip(pn_score, nn_score):
            if pn_d < 0 and nn_d < 0:
                score = 0
            elif pn_d < 0 and nn_d > 0:
                score = 40
            elif pn_d > 0 and nn_d < 0:
                score = 60
            elif pn_d > 0 and nn_d > 0:
                score = 100
            scores.append(score)

    elif graphType == "Fancy":
        cmap = mpl.cm.hot
        speed = np.array([i["Final speed"] for i in stats[indexes, 3]])
        time = stats[indexes, 2].astype("float64")
        scores = (900 - speed) / time
        scores[hit_or_miss == 0] = 0
        min_score, max_score = 0, 3

    elif graphType == "PN vs NN Fancy":
        min_score, max_score = 0, 500
        cmap = mpl.cm.gist_ncar

        pn_score = stats[indexes, 1].astype("float64")
        nn_score = stats[indexes_neural, 1].astype("float64")
        if not len(nn_score):
            return

        speed = np.array([i["Final speed"] for i in stats[indexes, 3]])
        time = stats[indexes, 2].astype("float64")
        scores_pn = (900 - speed) / time
        scores_pn[hit_or_miss == 0] = 0

        speed = np.array([i["Final speed"] for i in stats[indexes_neural, 3]])
        time = stats[indexes_neural, 2].astype("float64")
        scores_nn = (900 - speed) / time
        scores_nn[hit_or_miss == 0] = 0

        scores = []
        for i, (pn_d, nn_d) in enumerate(zip(pn_score, nn_score)):
            if pn_d < 0 and nn_d < 0:
                score = 500  # white
            elif pn_d < 0 and nn_d > 0:
                score = 370  # red
            elif pn_d > 0 and nn_d < 0:
                score = 90  # cyan
            elif pn_d > 0 and nn_d > 0:
                if scores_nn[i] > scores_pn[i]:
                    score = 300  # yellow
                else:
                    score = 50  # blue
            scores.append(score)

    if not (min_score or max_score):
        min_score, max_score = min(scores), max(scores)
    norm = mpl.colors.Normalize(vmin=min_score, vmax=max_score)

    nr = len(set(distances))#50
    ntheta = len(set(angles))#200
    r_edges = np.linspace(min(distances), max(distances), nr + 1)
    theta_edges = np.linspace(-np.pi/2, np.pi/2, ntheta + 1)
    H, _, _ = np.histogram2d(distances, np.deg2rad(angles), [r_edges, theta_edges], weights=scores)

    # Plot
    ax = plt.subplot(111, polar=True)
    Theta, R = np.meshgrid(theta_edges, r_edges)
    ax.pcolormesh(Theta, R, H, norm=norm, cmap=cmap)
    # plt.show()

    # im = ax.hist2d(angles, distances, weights=scores, norm=norm, cmap=cmap,
    #                bins=[len(set(angles)), len(set(distances))])

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cax, orientation='vertical')

    ax.set_theta_zero_location("N")
    ax.set_thetamin(90)
    ax.set_thetamax(-90)

    # z = 0

    if save_path:
        plt.savefig(f"{save_path}.png", format="png")
        plt.close()


def printAllGraphs(data):
    maneuvers, coefficients = set([infos["maneuver"] for infos in data[:, 0]]), \
                              set([infos["coefficient"] for infos in data[:, 0]])

    for maneuver, coefficient in product(maneuvers, coefficients):
        for graphName in drawAccuracy.ALL_NAMES:
            prefix = graphName.replace(" ", "_")
            file_name = f"graphs/big_boy/{prefix} maneuver={maneuver} coefficient={np.round(float(coefficient), 1)}"

            drawAccuracy(stats=data, maneuver=maneuver, coefficient=coefficient, graphType=graphName,
                         save_path=file_name)
