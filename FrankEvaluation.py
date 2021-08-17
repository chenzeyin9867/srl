import numpy as np
from a2c_ppo_acktr.FrankENV import *

def frenk_algorithm(file):
    testData = np.load(file, allow_pickle=True)
    distance = []
    delta_angle = []
    distance_none = []
    delta_angle_none = []
    for _ in range(100):
        # if _ % 10 == 0:
        #     print(_)
        # print(10 * "*")
        passive_haptics_env = FrankEnv(testData[_], _)
        x_l, y_l, x_v_flag, y_v_flag, x_p_flag, y_p_flag, tangent_x, tangent_y, dis, ang = \
            passive_haptics_env.eval()
