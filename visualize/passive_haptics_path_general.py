"""
This file is used to visualize the data generalization
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys
import math
from matplotlib.backends.backend_pdf import PdfPages
VELOCITY = 1.4 / 60.0
HEIGHT, WIDTH = 30, 30
FRAME_RATE = 60
PI = np.pi
STEP_LOW = int(0.5 / VELOCITY)
STEP_HIGH = int(3.5 / VELOCITY)

random.seed()

# whether outside the tracing space
def outbound(x, y):
    if x <= 0 or x >= WIDTH or y <= 0 or y >= HEIGHT:
        return True
    else:
        return False

# place the user in the target prop
def initialize():
    # dt = -PI + PI*random.random()*2
    dt = -PI    / 2
    xt = WIDTH  / 2
    yt = HEIGHT / 2
    return xt, yt, dt



# normalize the theta into [-PI,PI]
def norm(theta):
    if theta < -PI:
        theta = theta + 2 * PI
    elif theta > PI:
        theta = theta - 2 * PI
    return theta


if __name__ == '__main__':
    result = []
    len_ = []
    dir = "../Dataset/"+"path_general_w" + str(int(WIDTH)) + 'h' + str(int(HEIGHT)) 
    if not os.path.exists(dir):
        os.makedirs(dir)
    for Epoch in range(5000):
        if Epoch % 100 == 0:
            print("Epoch:", Epoch)
        x, y, d = initialize()
        init_x, init_y = x, y
        Xt = []
        Yt = []
        Dt = []
        Dchange = []
        Xt.append(x)
        Yt.append(y)
        Dt.append(norm(d+PI)) # inverse the angle
        Dchange.append(0)
        iter = 0
        delta_direction_per_iter = 0
        num_change_direction = 0
        turn_flag = 0
        for t in range(50000):
            if turn_flag == 0:
                turn_flag = np.random.randint(STEP_LOW, STEP_HIGH)
                delta_direction = random.normalvariate(0, 45)
                delta_direction = delta_direction * PI / 180.
                random_radius = 1 * random.random() + 1
                num_change_direction = abs(delta_direction * random_radius / VELOCITY)
                delta_direction_per_iter = delta_direction / num_change_direction

            if num_change_direction > 0:
                d = norm(d + delta_direction_per_iter) # update the direction
                num_change_direction = num_change_direction - 1
            else:
                turn_flag = turn_flag - 1
                delta_direction_per_iter = 0
            x = x + VELOCITY * np.cos(d)
            y = y + VELOCITY * np.sin(d)
            if outbound(x, y):
                break
            Xt.append(x)
            Yt.append(y)
            Dt.append(norm(d+PI))
            Dchange.append(-delta_direction_per_iter) #inverse
        Xt = Xt[::-1]
        Yt = Yt[::-1]
        Dt = Dt[::-1]
        Dchange = Dchange[::-1]

        Xt_np = np.array(Xt)
        Yt_np = np.array(Yt)
        Dt_np = np.array(Dt)
        Dchange_np = np.array(Dchange)
        stack_data = np.stack((Xt_np, Yt_np, Dt, Dchange_np), axis=-1)
        result.append(stack_data)
        len_.append(t)

        plt.figure(figsize=(5,5))
        if Epoch < 100:
            print(Epoch)
            # plt.axis([0.0, WIDTH, 0.0, HEIGHT])
            plt.axis('scaled')
            plt.xlim(0.0, WIDTH)
            plt.ylim(0.0, HEIGHT)
            plt.yticks([])
            plt.xticks([])

            # plt.axis('off')
            dst = str(Epoch) + '.pdf'
            dst1 = str(Epoch) + '.png'
            plt.scatter(Xt, Yt, s=0.5, c=[t for t in range(len(Xt))], cmap="Reds", alpha = 0.2)
            plt.scatter(init_x, init_y, c='gold', s=100, marker="*", label='target object', edgecolors='orange', linewidths=0.5)

            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)

            plt.legend()
            plt.savefig(dir + "/" + dst1)
            # pp = PdfPages(dir + "/" + dst)
            # plt.savefig('./Da/general/' + name[evalType]+ '_' + str(t) +'.png')
            # plt.cla()
            # pp.savefig()
            # pp.close()
            plt.close()
    save_np = np.array(result)
    np.save('../Dataset/eval_path_30_constrain.npy', save_np)
    print(np.mean(len_), np.std(len_))
    # sys.exit()