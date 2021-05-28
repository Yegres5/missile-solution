import pandas
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from math import floor
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

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
        x0 = dataLines[0][0,num]
        y0 = dataLines[0][1,num]
        z0 = dataLines[0][2,num]
        
        ax.set_xbound([-delta+x0, delta+x0])
        ax.set_ybound([-delta+y0, delta+y0])
        ax.set_zbound([-delta+z0, delta+z0])

        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        # if np.allclose(data, dataLines[-1])
    return lines


def drawAnimation():
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    #Turn off axis
    ax.set_axis_off()

    #Create axes x y z
    xspan, yspan, zspan = 3 * [np.linspace(0,20000,20)]
    zero = np.zeros_like(xspan)

    ax.plot3D(xspan, zero, zero, 'k-.', linewidth=0.8)
    ax.plot3D(zero, yspan, zero, 'k-.', linewidth=0.8)
    ax.plot3D(zero, zero, zspan, 'k-.', linewidth=0.8)

    ax.text(xspan.max() + 10, .5, .5, "Z", color='red', fontsize = 20)
    ax.text(.5, yspan.max() + 10, .5, "X", color='red', fontsize = 20)
    ax.text(.5, .5, zspan.max() + 10, "Y", color='red', fontsize = 20)

    # Getting CSV data
    column_names = ['x', 'y', 'z']
    list = []
    for i in range(2):
        list.append(pandas.read_csv("%d.csv" % (i), header=None))


    #Creating arrays of data from CSV
    data = []
    for la in list:
        data.append(np.array(la.values).transpose())

    # Creating fifty line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    lines = [ax.plot(dat[2, 0:1], dat[0, 0:1], dat[1, 0:1])[0] for dat in data]

    #X-Z Y-X Z-Y

    # Setting the axes properties
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    #Setting x0 y0 z0
    delta = 10000.
    x0 = 0
    y0 = 0
    z0 = 0

    ax.set_xbound([-delta + x0, delta + x0])
    ax.set_ybound([-delta + y0, delta + y0])
    ax.set_zbound([-delta + z0, delta + z0])

    speed = 1
    indexes = np.arange(floor(data[0].shape[1]/speed))*speed
    if indexes[-1] != data[0].shape[1]-1:
        indexes = np.hstack([indexes, np.array([data[0].shape[1]-1])])

    data = [d[:, indexes] for d in data]

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, data[0].shape[1], fargs=(data, lines, ax),
                                       interval=10, blit=False, repeat=False)

    plt.show(block=True)



def drawAccuracy(stats, maneuver, coefficient, graphType="Score nonbin"):
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
        scores = [1 if nn_d < pn_d else 0 for pn_d, nn_d in zip(pn_dist, nn_dist)]
    elif graphType == "PN vs NN Distance diff":
        pn_dist = np.array([i["Distance"] for i in stats[indexes, 3]])
        nn_dist = np.array([i["Distance"] for i in stats[indexes_neural, 3]])

        scores = nn_dist - pn_dist
        scores = 1 / scores
        # min_score, max_score = -2000, 2000

    elif graphType == "PN vs NN Accuracy":
        cmap = mpl.cm.brg

        pn_score = stats[indexes, 1].astype("float64")
        nn_score = stats[indexes_neural, 1].astype("float64")

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
        min_score, max_score = 0, 100

    elif graphType == "Fancy":
        cmap = mpl.cm.hot
        speed = np.array([i["Final speed"] for i in stats[indexes, 3]])
        time = stats[indexes, 2].astype("float64")
        scores = (900 - speed) / time
        scores[hit_or_miss == 0] = 0
        min_score, max_score = 0, 3

    elif graphType == "PN vs NN Fancy":
        cmap = mpl.cm.gist_ncar

        pn_score = stats[indexes, 1].astype("float64")
        nn_score = stats[indexes_neural, 1].astype("float64")

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

        min_score, max_score = 0, 500

    if not (min_score or max_score):
        min_score, max_score = min(scores), max(scores)

    norm = mpl.colors.Normalize(vmin=min_score, vmax=max_score)

    im = ax.hist2d(angles, distances, weights=scores, norm=norm, cmap=cmap,
                   bins=[len(set(angles)), len(set(distances))])

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cax, orientation='vertical')