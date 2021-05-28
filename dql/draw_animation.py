import pandas
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from math import floor


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