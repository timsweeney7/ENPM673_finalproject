
import json
import matplotlib.pyplot as plt
import numpy as np

# displays/saves data from json file
# first argument: path
# second argument: 0,1 - DISPLAY - shows all plots. DEFAULT: 1 [ON]
# thirds argument: 0,1 - SAVE - saves all images + summary in respective folder locations. DEFAULT: 0 [OFF]
# fourht arugment: 0,1 - TEMP - saves adll images + summary in alternate path under algorithm folder to avoid overwriting data
# NOTE: SAVE ON WILL OVERWRITE EXISITING FILES WITH SAME NAMES IN DESGINATED FOLDERS

def displayResults(path, display = 1, save = 0, temp = 1):

    subpath = path[:35]
    if temp:
        subpath = subpath + 'temp/'
    figpath = subpath + 'figs/'

    with open(path, 'r') as openfile:
        results = json.load(openfile)


    alg_name = results["algorithm description"]
    alg_name = alg_name[:12]
    alg_num = alg_name[10] 


    t1 = "==================================\nSUMMARY OF RESULTS\n=================================="
    t2 = f'Description: {results["algorithm description"]}'
    t3 = f'Mean time [s]: {results["mean time"]}'
    t4 = f'Total time [s]: {results["total time"]}'
    t5 = f'Start Pose [frame]: {results["start pose"]}'
    t6 = f'End Pose [frame]: {results["end pose"]}'

    print(t1)
    print(t2)
    print(t3)
    print(t4)
    print(t5)
    print(t6)
    

    # reading ground truth and estimated trajectory data
    odometery = np.array(results["odometry"])
    odometery = np.reshape(odometery, (-1, 3, 4))
    gt = np.array(results["ground truth"])
    gt = np.reshape(gt, (-1, 3, 4))

    # reading error data
    abserror = np.array(results["absolute error"])
    relerror = np.array(results["relative error"])
    angerror = np.array(results["angular heading error"])

    t7 = "Mean Abolute Error [m]: " + str(np.mean(abserror))
    t8 = "Mean Relative Error [m]: " + str(np.mean(relerror))
    t9 = "Mean Relative Angle Heading Error [deg]: " + str(np.mean(angerror))

    print(t7)
    print(t8)
    print(t9)



    # Plot ground truth
    fig = plt.figure(figsize=(14, 14))
    title1 = alg_name + ' Trajectory' 
    plt.title(title1)
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
    if display:
        plt.pause(1e-32) 


    fig2 = plt.figure(figsize=(10,10))
    ax2 = fig2.add_subplot()
    plt.title(alg_name + " Absolute Error")
    ax2.plot(range(len(abserror)),abserror)
    plt.xlabel("Frame")
    plt.ylabel("Absolute Error (meters)")
    if display:
        plt.waitforbuttonpress()

    fig3 = plt.figure(figsize=(10,10))
    ax3 = fig3.add_subplot()
    plt.title(alg_name + " Relative Error")
    ax3.plot(range(len(relerror)),relerror)
    plt.xlabel("Frame")
    plt.ylabel("Relative Error (meters)")
    if display:
        plt.waitforbuttonpress()

    fig4 = plt.figure(figsize=(10,10))
    ax4 = fig4.add_subplot()
    plt.title(alg_name + " Relative Heading Angle Error")
    ax4.plot(range(len(angerror)),angerror)
    plt.xlabel("Frame")
    plt.ylabel("Relative Heading Angle Error (degrees)")
    if display:
        plt.waitforbuttonpress()

    if save:
        path1 = figpath + 'agl_' + str(alg_num) + '_trajectory_.png'
        fig.savefig(path1)
        path2 = figpath + 'agl_' + str(alg_num) + '_absolute_error.png'
        fig2.savefig(path2)
        path3 = figpath + 'agl_' + str(alg_num) + '_relative_error.png'
        fig3.savefig(path3)
        path4 = figpath + 'agl_' + str(alg_num) + '_angular_heading_error.png'
        fig4.savefig(path4)

        textpath = subpath + 'alg_' + str(alg_num) + "_summary.txt"
        with open(textpath, 'w') as f:
            f.write('\n' + t1)
            f.write('\n' + t2)
            f.write('\n' + t3)
            f.write('\n' + t4)
            f.write('\n' + t5)
            f.write('\n' + t6)
            f.write('\n' + t7)
            f.write('\n' + t8)
            f.write('\n' + t9)

    return 0


# apparently need this to import other functions from here into other scripts
# to call this script alone, change path appropriatley
def main():

    path = "./kittiDataSet/results/algorithm_1/algorithm_1.json"
    displayResults(path)
    return 0


if __name__ == '__main__':
    main()



