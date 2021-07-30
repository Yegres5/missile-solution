import matplotlib.pyplot as plt
import numpy as np


def graph(rocket_log, la_log, true_overload):
    true_overload = np.array(true_overload)

    rocket_coord = rocket_log[:, 0:3]
    la_coord = la_log[:, 0:3]
    np.savetxt("0.csv", rocket_coord[:, [2, 0, 1]], delimiter=",")
    np.savetxt("1.csv", la_coord[:, [2, 0, 1]], delimiter=",")

    overloads = rocket_log[:, 10:12]

    plt.figure(figsize=[16, 9])

    ax1 = plt.subplot(321)
    ax1.plot(rocket_coord.torch[0], rocket_coord.torch[1])
    ax1.plot(la_coord.torch[0], la_coord.torch[1])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = plt.subplot(322)
    ax2.plot(rocket_coord.torch[0], rocket_coord.torch[2])
    ax2.plot(la_coord.torch[0], la_coord.torch[2])
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    ax3 = plt.subplot(323)
    ax3.plot(rocket_coord.torch[2], rocket_coord.torch[1])
    ax3.plot(la_coord.torch[2], la_coord.torch[1])
    ax3.set_xlabel("Z")
    ax3.set_ylabel("Y")

    ax4 = plt.subplot(325)
    ax4.plot(overloads.torch[0], label="Nz", color="green")
    ax4.plot(true_overload.T[0], label="True Nz", linestyle='dashed', color="green")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Overload")
    ax4.legend()

    ax5 = plt.subplot(326)
    ax5.plot(overloads.torch[1], label="Ny", color="blue")
    ax5.plot(true_overload.T[1], label="True Ny", linestyle='dashed', color="blue")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Overload")
    ax5.legend()

    plt.show(block=True)
