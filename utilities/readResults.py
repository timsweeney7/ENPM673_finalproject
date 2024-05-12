
import json
import matplotlib.pyplot as plt
import numpy as np


"""
Script to read results JSON files and display them
"""

# funciton to be used
def displayResults(path):

    with open(path, 'r') as openfile:
        results = json.load(openfile)
    
    print("==================================\nSUMMARY OF RESULTS")
    print(f'Description: \n{results["algorithm description"]}')
    print(f'Mean time [s]: {results["mean time"]}')
    print(f'Total time [s]: {results["total time"]}')
    print(f'Start Pose [frame]: {results["start pose"]}')
    print(f'End Pose [frame]: {results["end pose"]}')
    

    # reading ground truth and estimated trajectory data
    odometery = np.array(results["odometry"])
    odometery = np.reshape(odometery, (-1, 3, 4))
    gt = np.array(results["ground truth"])
    gt = np.reshape(gt, (-1, 3, 4))

    # reading error data
    abserror = np.array(results["absolute error"])
    relerror = np.array(results["relative error"])
    angerror = np.array(results["angular heading error"])

    print("Mean Abolute Error [m]: ", np.mean(abserror))
    print("Mean Relative Error [m]: ", np.mean(relerror))
    print("Mean Relative Angle Heading Error [deg]: ", np.mean(angerror))



    # Plot ground truth
    fig = plt.figure(figsize=(14, 14))
    plt.title("Trajectory")
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-20, azim=270)
    xs = gt[:, 0, 3]
    ys = gt[:, 1, 3]
    zs = gt[:, 2, 3]
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.plot(xs, ys, zs, c='b')

    # Plot estimated trajectory
    xs = odometery[:, 0, 3]
    ys = odometery[:, 1, 3]
    zs = odometery[:, 2, 3]
    plt.plot(xs, ys, zs, c='r')
    plt.pause(1e-32)

    fig2 = plt.figure(figsize=(10,10))
    ax2 = fig2.add_subplot()
    plt.title("Absolute Error")
    ax2.plot(range(len(abserror)),abserror)
    plt.xlabel("Frame")
    plt.ylabel("Absolute Error (meters)")
    plt.waitforbuttonpress()

    fig3 = plt.figure(figsize=(10,10))
    ax3 = fig3.add_subplot()
    plt.title("Relative Error")
    ax3.plot(range(len(relerror)),relerror)
    plt.xlabel("Frame")
    plt.ylabel("Relative Error (meters)")
    plt.waitforbuttonpress()

    fig4 = plt.figure(figsize=(10,10))
    ax4 = fig4.add_subplot()
    plt.title("Relative Heading Angle Error")
    ax4.plot(range(len(angerror)),angerror)
    plt.xlabel("Frame")
    plt.ylabel("Relative Heading Angle Error (degrees)")
    plt.waitforbuttonpress()




    return 0


# apparently need this to import other functions from here into other scripts
# to call this script alone, uncomment function call and define path
def main():

    path = "./kittiDataSet/results/algorithm_6.json"
    displayResults(path)
    return 0


if __name__ == '__main__':
    main()





'''

path = "./kittiDataSet/results/algorithm_1.json"

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

'''