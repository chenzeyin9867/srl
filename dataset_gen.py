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
FRAME_RATE = 60
PI = np.pi
STEP_LOW = int(0.5 / VELOCITY)
STEP_HIGH = int(3.5 / VELOCITY)

random.seed()

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
    dir = "dataset"
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    
    pathnum = 5000 if mode == 'eval' else 100000
    for Epoch in trange(pathnum):
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
                random_radius = 2 * random.random() + 1
                num_change_direction = abs(delta_direction * random_radius / VELOCITY)
                delta_direction_per_iter = delta_direction / num_change_direction

            if num_change_direction > 0:
                num_change_direction = num_change_direction - 1
            else:
                turn_flag = turn_flag - 1
                delta_direction_per_iter = 0

            Dchange.append(delta_direction_per_iter)
        Dchange_np = np.array(Dchange)
        result.append(Dchange)
        len_.append(t)



    save_np = np.array(result)
    save_path = os.path.join(dir, mode+".npy")
    np.save(save_path, save_np)
    print(np.mean(len_), np.std(len_))
