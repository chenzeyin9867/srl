import numpy as np
import random
import math
HEIGHT, WIDTH = 8, 8
savetxt = 'eval_position.txt'
x_l = []
y_l = []
for i in range(5000):
    x = WIDTH * random.random()
    y = HEIGHT * random.random()
    x_l.append(x)
    y_l.append(y)

x_np = np.array(x_l).reshape((-1,1))
y_np = np.array(y_l).reshape((-1,1))
xy = np.concatenate((x_np, y_np),1)
np.savetxt(savetxt, xy)