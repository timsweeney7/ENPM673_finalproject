import json
import matplotlib.pyplot as plt
import numpy as np

path = "./kittiDataSet/results/test_output.json"

if __name__ == "__main__":
    
    with open(path, 'r') as openfile:
        results = json.load(openfile)
    
    print(f'Mean time: {results["mean time"]}')
    print(f'Total time: {results["total time"]}')

    odometery = np.array(results["odometry"])
    odometery = np.reshape(odometery, (-1, 3, 4))
    gt = np.array(results["ground truth"])
    gt = np.reshape(gt, (-1, 3, 4))

    # Plot ground truth
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-20, azim=270)
    xs = gt[:, 0, 3]
    ys = gt[:, 1, 3]
    zs = gt[:, 2, 3]
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.plot(xs, ys, zs, c='b')

    # Plot results
    xs = odometery[:, 0, 3]
    ys = odometery[:, 1, 3]
    zs = odometery[:, 2, 3]
    plt.plot(xs, ys, zs, c='r')
    plt.pause(1e-32)

    plt.waitforbuttonpress()