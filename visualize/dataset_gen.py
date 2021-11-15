"""
This file is used to visualize the data generalization
"""
from genericpath import exists
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys
import math
import argparse
import tqdm
from tqdm import trange
from matplotlib.backends.backend_pdf import PdfPages

VELOCITY = 1.4 / 60.0
HEIGHT, WIDTH = 15, 15
FRAME_RATE = 60
PI = np.pi
STEP_LOW = int(0.5 / VELOCITY)
STEP_HIGH = int(3.5 / VELOCITY)

random.seed()

"""
four seeds represent 4 initial location
"""
def initialize():
    xt = WIDTH / 2
    yt = HEIGHT / 2
    dt = PI/4
    return xt, yt, dt


"""
normalize the theta into [-PI,PI]
"""
def norm(theta):
    if theta < -PI:
        theta = theta + 2 * PI
    elif theta > PI:
        theta = theta - 2 * PI
    return theta


parser = argparse.ArgumentParser(description="training path generation.")
parser.add_argument('--mode', type=str, default='train', help='training or evaluation')
args = parser.parse_args()
print(args)
mode = args.mode


if __name__ == '__main__':
    result = []
    len_ = []
    dir = "../Dataset/"+"path_h_" + str(int(HEIGHT))+'w_'+str(int(WIDTH))
    if mode == 'train':
        dir += '_train'
    else:
        dir += '_eval'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    
    pathnum = 5000 if mode == 'eval' else 100000
    for Epoch in trange(pathnum):
        # if Epoch % 1000 == 0:
        #     print("Epoch:", Epoch)
        Dchange = []
        Dchange.append(0)
        iter = 0
        delta_direction_per_iter = 0
        num_change_direction = 0
        turn_flag = 0
        for t in range(3600):
            if turn_flag == 0:
                turn_flag = np.random.randint(STEP_LOW, STEP_HIGH)
                delta_direction = random.normalvariate(0, 45)
                delta_direction = delta_direction * PI / 180.
                random_radius = 1 * random.random() + 2
                num_change_direction = abs(delta_direction * random_radius / VELOCITY)
                # print(delta_direction * 180 / PI, num_change_direction)
                delta_direction_per_iter = delta_direction / num_change_direction
                # print("delta_per:", delta_direction_per_iter)

            if num_change_direction > 0:
                num_change_direction = num_change_direction - 1
            else:
                turn_flag = turn_flag - 1
                delta_direction_per_iter = 0

            Dchange.append(delta_direction_per_iter)
            # print(-delta_direction_per_iter)
        # Xt = Xt[::-1]
        # Yt = Yt[::-1]
        # Dt = Dt[::-1]
        # Dchange = Dchange[::-]
        #     # print(D_t * 180 / np.pi)
        #     # plt.axis((0, 0, 20, 20))
        # #     plt.scatter(X_t,Y_t,c = 'r',s = 0.1)
        # #     plt.pause(0.1)
        # # plt.show()
        # Xt_np = np.array(Xt)
        # Yt_np = np.array(Yt)
        # Dt_np = np.array(Dt)
        Dchange_np = np.array(Dchange)
        # stack_data = np.stack((Xt_np, Yt_np, Dt, Dchange_np), axis=-1)
        result.append(Dchange)
        len_.append(t)



    save_np = np.array(result)
    save_dir = os.path.join("../Dataset/Train_", mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, str(HEIGHT) + ".npy"), save_np)
    print(np.mean(len_), np.std(len_))
    # sys.exit() 