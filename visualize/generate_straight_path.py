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
HEIGHT, WIDTH = 10, 10
FRAME_RATE = 60
PI = np.pi
STEP_LOW = int(0.5 / VELOCITY)
STEP_HIGH = int(3.5 / VELOCITY)

random.seed()


def outbound(x, y):
    if x <= 0 or x >= WIDTH or y <= 0 or y >= HEIGHT:
        return True
    else:
        return False


"""
four seeds represent 4 initial location
"""


def initialize(seed):
    # xt = WIDTH/(2.5) + 0.2 * WIDTH * random.random()
    # yt = HEIGHT/(2.5) + 0.2 * HEIGHT * random.random()
    dt = -PI + PI*random.random()*2
    xt = WIDTH /2
    yt = HEIGHT / 2
    # dt = PI/4
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


if __name__ == '__main__':
    result = []
    len_ = []
    dir = "../Dataset/"+"path_straight" + str(int(WIDTH))
    if not os.path.exists(dir):
        os.makedirs(dir)
    for Epoch in range(500):
        if Epoch % 100 == 0:
            print("Epoch:", Epoch)
        seed = random.randint(1, 4)
        x, y, d = initialize(seed)
        init_x, init_y = x, y
        Xt = []
        Yt = []
        Dt = []
        Dchange = []
        Xt.append(x)
        Yt.append(y)
        Dt.append(norm(d+PI))
        Dchange.append(0)
        iter = 0
        delta_direction_per_iter = 0
        num_change_direction = 0
        turn_flag = 0
        for t in range(50000):
            if turn_flag == 0:
                turn_flag = np.random.randint(STEP_LOW, STEP_HIGH)
                delta_direction = 0
                random_radius = 2 * random.random() + 2
                num_change_direction = abs(delta_direction * random_radius / VELOCITY)
                # print(delta_direction * 180 / PI, num_change_direction)
                # delta_direction_per_iter = delta_direction / num_change_direction
                # print("delta_per:", delta_direction_per_iter)

            # if num_change_direction > 0:
            #     d = norm(d + delta_direction_per_iter)
            #
            #     num_change_direction = num_change_direction - 1
            # else:
                turn_flag = turn_flag - 1
                delta_direction_per_iter = 0
            x = x + VELOCITY * np.cos(d)
            y = y + VELOCITY * np.sin(d)
            if outbound(x, y):
                break
            Xt.append(x)
            Yt.append(y)
            Dt.append(norm(d+PI))
            Dchange.append(-delta_direction_per_iter)
            # print(-delta_direction_per_iter)
        Xt = Xt[::-1]
        Yt = Yt[::-1]
        Dt = Dt[::-1]
        Dchange = Dchange[::-1]
            # print(D_t * 180 / np.pi)
            # plt.axis((0, 0, 20, 20))
        #     plt.scatter(X_t,Y_t,c = 'r',s = 0.1)
        #     plt.pause(0.1)
        # plt.show()
        Xt_np = np.array(Xt)
        Yt_np = np.array(Yt)
        Dt_np = np.array(Dt)
        Dchange_np = np.array(Dchange)
        stack_data = np.stack((Xt_np, Yt_np, Dt, Dchange_np), axis=-1)
        result.append(stack_data)
        len_.append(t)
        #
        plt.figure(figsize=(5,5))
        if Epoch < 100:
            print(Epoch)
            # plt.axis([0.0, WIDTH, 0.0, HEIGHT])
            plt.axis('scaled')
            plt.xlim(0.0, WIDTH)
            plt.ylim(0.0, HEIGHT)
            # plt.yticks([])
            # plt.xticks([])

            # plt.axis('off')
            dst = str(Epoch) + '.png'
            # for i in range(len(Xt)):
            #     #     # print(float(i/len(x)))
            #     plt.scatter(np.array(Xt[i]), np.array(Yt[i]), s=1, c='r',
            #                     alpha=1.0 * math.exp(5 * (i / len(Xt) - 1.0)))
            plt.scatter(Xt, Yt, s=0.5, c=[t for t in range(len(Xt))], cmap="Blues", alpha = 0.2)
            # plt.scatter(Xt, Yt, c='r', s=1)
            plt.scatter(init_x, init_y, c='gold', s=100, marker="*", label='target object', edgecolors='orange', linewidths=0.5)
            plt.legend()
            plt.savefig(dir + "/" + dst)
            # pp = PdfPages(dir + "/" + dst)
            # plt.savefig('./Da/general/' + name[evalType]+ '_' + str(t) +'.png')
            plt.cla()
            # pp.savefig()
            # pp.close()
            plt.close()
        # plt.savefig("../Dataset/virtual_path_red/" + dst)

    save_np = np.array(result)
    np.save('../Dataset/eval_path_10_straight.npy', save_np)
    print(np.mean(len_), np.std(len_))
    sys.exit()