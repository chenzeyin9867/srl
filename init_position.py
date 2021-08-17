import numpy as np
HEIGHT, WIDTH = 16, 24

def init_eval_state(ind, evalType=0):
    if evalType == 2:
        m = (int(ind / 30 )) % 4
        n = (ind % 30) / 30.0
        # n = np.random.random()
        if m == 0:
            x_physical = 0
            y_physical = HEIGHT * n
            # self.p_direction = 0
        elif m == 1:
            x_physical = WIDTH * n
            y_physical = HEIGHT
            # self.p_direction = -PI / 2
        elif m == 2:
            x_physical = WIDTH
            y_physical = HEIGHT * n
            # self.p_direction = -PI
        elif m == 3:
            x_physical = WIDTH * n
            y_physical = 0
            # self.p_direction = PI / 2
    # self.print()
    # print(self.p_direction)
    return x_physical, y_physical

if __name__ == "__main__":
    result = []
    for t in range(5000):
        x, y = init_eval_state(t, 2)
        result.append([x,y])
    r = np.array(result)
    np.savetxt('position20x30.txt', r)
