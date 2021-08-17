import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from a2c_ppo_acktr.distributions import FixedNormal
import random
import matplotlib.pyplot as plt
import math
OBS_NORM = False
REWARD_SCALE = False
VELOCITY = 1.4 / 60.0
# HEIGHT, WIDTH = 12.0, 12.0
# HEIGHT_ALL, WIDTH_ALL = 15.0, 15.0
# HEIGHT, WIDTH = 16.0, 16.0
# HEIGHT_ALL, WIDTH_ALL = 20.0, 20.0
# HEIGHT, WIDTH = 8.0, 8.0
# HEIGHT_ALL, WIDTH_ALL = 10.0, 10.0
HEIGHT, WIDTH = 24 , 24
HEIGHT_ALL, WIDTH_ALL = 30.0, 30.0
# HEIGHT, WIDTH = 5.79, 5.79
# HEIGHT_ALL, WIDTH_ALL = 8.0, 8.0
FRAME_RATE = 60
RATIO = 10
PI = np.pi
STEP_LOW = int(0.5 / VELOCITY)
STEP_HIGH = int(3.5 / VELOCITY)
DELTA_X = (WIDTH_ALL - WIDTH) / 2.0
DELTA_Y = (HEIGHT_ALL - HEIGHT) / 2.0
PENALTY = torch.Tensor([1.0])
OBSERVATION_SPACE = 6


"""
This is the rdw environment
"""


def split(action):
    a, b, c = action[0]
    # a = torch.Tensor([1.0])
    # # b = torch.Tensor([1.0])
    # b = 0.2 + 2 * (b + 1.) / 2.
    # a = 0.2 + 2 * (a + 1.) / 2.
    # c = -0.5 + 1.0 * (c + 1.) / 2.
    # b = 0.2 + 2 * (b + 1.) / 2.
    # a = 0.2 + 2 * (a + 1.) / 2.
    # c = -0.5 + 1.0 * (c + 1.) / 2.
    # b = 0.5 + 1 * (b + 1.) / 2.
    # a = 0.5 + 1 * (a + 1.) / 2.
    # c = -0.45 + 0.9 * (c + 1.) / 2.
    # b = torch.clamp(1 + b, 0.5, 1.5)
    # a = torch.clamp(1 + a, 0.5, 1.5)
    # c = torch.clamp(c, -0.5, 0.5)
    # b = torch.Tensor([1.0])
    b = torch.clamp(b+1, 0.8, 1.49)
    a = torch.clamp(a+1, 0.86, 1.26)
    c = torch.clamp(c, -0.13, 0.13)
    # a = torch.Tensor([1.26])
    # b = torch.Tensor([1.49])
    # c = torch.Tensor([0])
    # a = torch.clamp(0.86 + 0.4 * (a + 1.) / 2., 0.86, 1.26)
    # b = torch.clamp(0.8 + 0.69 * (b + 1.) / 2., 0.8, 1.49)
    # c = torch.clamp(c, -0.13, 0.13)
    #
    # a = 0.86 + 0.4 * (a + 1.) / 2.
    # b = 0.8 + 0.69 * (b + 1.) / 2.
    # c = -0.13 + 0.26 * (c + 1.) / 2.
    return a, b, c


def initialize(seed):
    xt, yt, dt = 0, 0, 0
    if seed == 1:
        xt = 0
        yt = random.random() * HEIGHT
        # dt = random.random() * PI - PI / 2.
        dt = 0
    elif seed == 2:
        xt = random.random() * WIDTH
        yt = HEIGHT
        # dt = -random.random() * PI
        dt = -PI / 2.0
    elif seed == 3:
        xt = WIDTH
        yt = random.random() * HEIGHT
        # dt = random.randint(0, 1) * 1.5 * PI + random.random() * PI / 2. - PI
        dt = -PI
    elif seed == 4:
        xt = random.random() * WIDTH
        yt = 0
        # dt = random.random() * PI
        dt = PI / 2
    # xt = random.random() * WIDTH
    # yt = random.random() * HEIGHT
    # dt = random.random() * PI * 2 - PI

    return xt, yt, dt




def norm(theta):
    if theta < -PI:
        theta = theta + 2 * PI
    elif theta > PI:
        theta = theta - 2 * PI
    return theta


def outbound(x, y):
    if x <= 0 or x >= WIDTH or y <= 0 or y >= HEIGHT:
        return True
    else:
        return False


def min_length_direction(x, y, a, b, cos):  # cause the heading has the direction
    p1 = torch.Tensor([0, b])
    # p2 = np.array([1,a+b])
    p2 = torch.Tensor([1, a + b])
    # p3 = np.array([-b/a,0])
    p3 = torch.Tensor([-b / a, 0])
    # p4 =np.array([(1-b)/a,1])
    p4 = torch.Tensor([(1 - b) / a, 1.])
    p = torch.cat((p1, p2, p3, p4))
    # p = np.concatenate((p1,p2,p3,p4),axis=0)
    p = p.reshape((4, 2))
    p = p[p[:, 0].argsort(), :]
    if cos > 0:
        c, d = p[2]
    else:
        c, d = p[1]
    len = distance(x, y, c, d)
    # len = min(distance(x, y, c, d), distance(x, y, e, f))
    return len


def min_length(x, y, a, b):  # min length of the line y = ax+b with intersection with the bounding box of [0,1]
    p1 = torch.Tensor([0, b])
    # p2 = np.array([1,a+b])
    p2 = torch.Tensor([1, a + b])
    # p3 = np.array([-b/a,0])
    p3 = torch.Tensor([-b / a, 0])
    # p4 =np.array([(1-b)/a,1])
    p4 = torch.Tensor([(1 - b) / a, 1.])
    p = torch.cat((p1, p2, p3, p4))
    # p = np.concatenate((p1,p2,p3,p4),axis=0)
    p = p.reshape((4, 2))
    p = p[p[:, 0].argsort(), :]
    c, d = p[1]
    e, f = p[2]
    return min(distance(x, y, c, d), distance(x, y, e, f))


def distance(x, y, a, b):
    # return np.sqrt(np.square(x-a)+np.square(y-b))
    # return torch.sqrt((x - a).pow(2) + (y - b).pow(2))
    return math.sqrt((x - a) * (x - a) + (y - b) * (y - b))


def toTensor(x):
    return torch.Tensor(x)

class PassiveHapticsEnv(object):
    def __init__(self, gamma, num_frame_stack, random=False, eval=False):
        self.r_l= []
        self.eval = eval
        self.distance = []
        self.num_frame_stack = num_frame_stack  # num of frames stcked before input to the MLP
        self.gamma = gamma                      # γ used in PPO
        self.observation_space = Box(-1., 1., (num_frame_stack * OBSERVATION_SPACE,))
        self.action_space = Box(-1.0, 1.0, (3,))
        self.obs = []
        self.v_direction = 0            # virtual direction of user
        self.p_direction = 0            # physical direction
        self.obj_x = WIDTH_ALL / 2      # target prop in v-space
        self.obj_y = HEIGHT_ALL / 2
        self.obj_x_p = WIDTH / 2
        self.obj_y_p = HEIGHT / 2       # target prop in p-space
        self.obj_d_p = PI / 4.0
        self.obj_d = PI / 4.0
        self.obj = 0
        self.x_physical = 0
        self.y_physical = 0
        self.x_virtual = 0
        self.y_virtual = 0
        self.time_step = 0
        self.direction_change_num = 0
        self.delta_direction_per_iter = 0
        self.current_obs = 0
        self.reward = torch.Tensor([0.0])
        self.path_cnt = 0
        # self.RS = []    # used for the reward-scaling
        self.r1=[]
        self.r2=[]
        self.r3=[]

        if not eval:
            self.pas_path_file = np.load('./Dataset/train_path_30.npy', allow_pickle=True)
            # print("Loading the training dataset.")
        else:
            self.pas_path_file = np.load('./Dataset/eval_path_30.npy', allow_pickle=True)
            # print("Loading the eval dataset")
        self.v_path = self.pas_path_file[self.path_cnt]
        self.v_step_pointer = 0  # the v_path counter


    '''
    This initial another kind of state, which put on the same edge as the virtual space
    '''
    #
    # def initialize_same_edge(seed):
    #     ratio = WIDTH / WIDTH_ALL
    #     if abs(self.x_virtual) < 0.1:
    #         self.x_physical = 0
    #         self.y_physical = self.y_virtual * ratio
    #         self.p_direction = 0
    #     elif abs(self.y_virtual - HEIGHT_ALL) < 0.1:
    #         self.x_physical = self.x_virtual * ratio
    #         self.y_physical = HEIGHT
    #         self.p_direction = -PI / 2
    #     elif abs(self.x_virtual - WIDTH_ALL) < 0.1:
    #         self.x_physical = WIDTH
    #         self.y_physical = self.y_virtual * ratio
    #         self.p_direction = -PI
    #     elif abs(self.y_virtual) < 0.1:
    #         self.x_physical = self.x_virtual * ratio
    #         self.y_physical = 0
    #         self.p_direction = PI / 2
    #     # self.p_direction = self.v_direction


    def get_obs(self):
        """
        scale the physical and virtual space into the same square box inside [-1,1]
        """
        if self.delta_direction_per_iter < 0:
            eye = -1.0
        elif self.delta_direction_per_iter > 0:
            eye = 1.0
        else:
            eye = 0.0
        # return [(self.x_physical+DELTA_X-(WIDTH_ALL)/2)/(WIDTH_ALL/2), (self.y_physical+DELTA_Y-(HEIGHT_ALL/2))/(HEIGHT_ALL/2), (self.p_direction) / (PI),
        #         (self.x_virtual-(WIDTH_ALL)/2)/(WIDTH_ALL/2), (self.y_virtual-(HEIGHT_ALL/2))/(HEIGHT_ALL/2), (self.v_direction) / (PI),
        #         (self.obj_x_p+DELTA_X-(WIDTH_ALL)/2)/(WIDTH_ALL/2), (self.obj_y_p+DELTA_Y-(HEIGHT_ALL)/2)/(HEIGHT_ALL/2), (self.obj_d_p)/(PI),
        #         (self.obj_x-(WIDTH_ALL/2))/(WIDTH_ALL/2), (self.obj_y-(HEIGHT_ALL/2))/(HEIGHT_ALL/2), (self.obj_d) / (PI)
        #         # ,eye
        #         ]
        return [(self.x_physical+DELTA_X-(WIDTH_ALL)/2)/(WIDTH_ALL/2), (self.y_physical+DELTA_Y-(HEIGHT_ALL/2))/(HEIGHT_ALL/2), (self.p_direction) / (PI),
                (self.x_virtual-(WIDTH_ALL)/2)/(WIDTH_ALL/2), (self.y_virtual-(HEIGHT_ALL/2))/(HEIGHT_ALL/2), (self.v_direction) / (PI)
                # ,eye
                # ,WIDTH,HEIGHT,
                # WIDTH_ALL, HEIGHT_ALL
                ]
        # return [(self.x_physical+DELTA_X-self.obj_x_p)/(WIDTH_ALL/2), (self.y_physical+DELTA_Y-self.obj_y_p)/(HEIGHT_ALL/2), (self.p_direction) / (PI),
        #         (self.x_virtual-self.obj_x)/(WIDTH_ALL/2), (self.y_virtual-self.obj_y)/(HEIGHT_ALL/2), (self.v_direction) / (PI)
        #         # ,eye
        #         ]

         # return [self.x_physical / WIDTH, self.y_physical / WIDTH, (self.p_direction + PI) / (2 * PI),
         #    self.x_virtual / WIDTH_ALL, self.y_virtual / HEIGHT_ALL, (self.v_direction + PI) / (2 * PI),
         #    self.obj_x / WIDTH_ALL, self.obj_y / HEIGHT_ALL, (self.obj_d + PI) / (2 * PI)]

    def reset(self):
        self.obs = []
        self.v_path = self.pas_path_file[self.path_cnt]
        self.path_cnt += 1  # next v_path
        self.path_cnt = self.path_cnt % len(self.pas_path_file)  # using the training data iteratively
        self.v_direction = 0
        self.p_direction = 0
        # if not self.eval:
        #     self.v_step_pointer = np.random.randint(0, len(self.v_path)-1)
        # else:
        #     self.v_step_pointer = 0
        self.v_step_pointer = 0
        self.time_step = 0
        self.direction_change_num = 0
        self.delta_direction_per_iter = 0
        seed = random.randint(1, 4)
        self.x_virtual, self.y_virtual, self.v_direction, self.delta_direction_per_iter \
            = self.v_path[self.v_step_pointer]                                      # initialize the user in virtual space
        x, y, d = initialize(seed)
        self.obj_x, self.obj_y, _, _ = self.v_path[-1]                     # initialize the target prop in virtual space
        # self.obj_d = norm(self.obj_d)
        self.x_physical = x
        self.y_physical = y
        self.p_direction = d
        # self.p_direction = self.compute_dir_to_obj() 
        init_obs = []
        init_obs.extend(10 * self.get_obs())
        self.obs.append(init_obs)
        self.obs = torch.Tensor(self.obs)
        self.current_obs = init_obs
        self.reward = torch.Tensor([0.0])
        self.r1 = []
        self.r2 = []
        self.r3 = []
        return self.obs

    def step(self, action):
        """
        step forward for K times using the same action based on action-repeat strategy.
        """
        gt, gr, gc = split(action)
        # print(gt.item(), gr.item(), gc.item())
        k = random.randint(5, 15)  # action repetition
        reward = torch.Tensor([0.0])  # initial reward for this step period
        for ep in range(k):  # for every iter, get the virtual path info, and steering
            self.vPathUpdate()
            signal = self.physical_step(gt, gr, gc)  # steering the physical env using the actions
            self.time_step += 1
            if (not signal) or (self.v_step_pointer == len(self.v_path) - 1):  # collision with the tracking space or finish the virtual path
                # reward += self.get_reward()
                if not signal:  # collide with the wall
                    reward = -PENALTY
                    # print(reward)
                break # when leave the tracking space or finish one path, end this episode
            elif ep == 0:
                reward = self.get_reward()

        obs = self.current_obs[OBSERVATION_SPACE:]  # update the observation after k steps 
        obs.extend(self.get_obs())
        self.current_obs = obs
        obs = toTensor(obs)

        # if REWARD_SCALE:
        #     mul = self.RS[-1] if len(self.RS) >= 1 else 0
        #     self.RS.append(reward + self.gamma * mul)
        #     if len(self.RS) > 1:
        #         div = np.std(self.RS) + 10e-5
        #         reward = (reward) / div
        ret_reward = reward
        self.reward += reward

        if not signal:  # reset the env when leave the tracking space
            bad_mask = 1
            r_reward = self.reward
            # print(distance(self.x_physical, self.y_physical, self.obj_x_p, self.obj_y_p))
            # print("r1:", sum(self.r1)/len(self.r1), "std:", np.std(self.r1), end='\t')
            # print("r2:", (sum(self.r2)/len(self.r2)).item(), "std:", np.std(self.r2), end='\t')
            # print("r3:", (sum(self.r3)/len(self.r3)), "std:", np.std(self.r3))
            # print("r1 - 3r2-r3/100:", (np.mean(self.r1) - 3 * np.mean(self.r2) - np.mean(self.r3))/100)
            # plt.plot(self.distance, self.r_l)
            # plt.show()
            # plt.cla()
            self.reset()
            # print(self.path_cnt)
            return obs, ret_reward, [1], [bad_mask], r_reward
        elif signal and self.v_step_pointer >= len(self.v_path) - 1: # successfully end one episode, get the final reward
            r_reward = self.reward
            # print(ret_reward, distance(self.x_physical, self.y_physical, self.obj_x_p, self.obj_y_p)/10)
            # ret_reward = (1 - 0.1*distance(self.x_physical, self.y_physical, self.obj_x_p, self.obj_y_p))
            ret_reward += self.final_reward()
            # print("distance:", distance(self.x_physical, self.y_physical, self.obj_x_p, self.obj_y_p))
            # print("r1:", sum(self.r1)/len(self.r1), "std:", np.std(self.r1), end='\t')
            # print("r2:", (sum(self.r2)/len(self.r2)).item(), "std:", np.std(self.r2), end='\t')
            # print("r3:", (sum(self.r3)/len(self.r3)), "std:", np.std(self.r3))
            # print("1-5r2-r3/100:", (1 - 5 * np.mean(self.r2) - np.mean(self.r3))/100)
            # plt.plot(self.distance, self.r_l)
            # plt.show()
            # plt.cla()
            self.reset()
            # print(self.path_cnt)
            return obs, ret_reward, [1], [0], r_reward
        else:
            return obs, ret_reward, [0], [0], ret_reward

    def vPathUpdate(self):
        self.x_virtual, self.y_virtual, self.v_direction, self.delta_direction_per_iter = \
            self.v_path[self.v_step_pointer]  # unpack the next timestep virtual value
        # self.delta_direction_per_iter = 0.0
        self.v_step_pointer += 1

    def physical_step(self, gt, gr, gc):
        delta_curvature = gc * (VELOCITY / gt)
        delta_rotation = self.delta_direction_per_iter / gr
        # delta_angle = 0
        # if abs(delta_curvature) > abs(delta_rotation):
        #     delta_angle = delta_curvature
        # else:
        #     delta_angle = delta_rotation
        # print(gt, gr, gc, delta_curvature, delta_rotation)
        delta_dis = VELOCITY / gt
        self.x_physical = self.x_physical + torch.cos(torch.Tensor([self.p_direction])) * delta_dis
        self.y_physical = self.y_physical + torch.sin(torch.Tensor([self.p_direction])) * delta_dis
        if outbound(self.x_physical, self.y_physical):
            return False


        self.p_direction = norm(self.p_direction + delta_curvature + delta_rotation)
        # self.p_direction = norm(self.p_direction + delta_angle)
        return True
        # delta_dis = VELOCITY



    '''
    when collide with the bundary in eval type, the user just reset instead of end the episode
    '''
    def physical_step_eval(self, gt, gr, gc):
        delta_curvature = gc * (VELOCITY / gt)
        delta_rotation = self.delta_direction_per_iter / gr
        delta_angle = 0
        if abs(delta_curvature) > abs(delta_rotation):
            delta_angle = delta_curvature
        else:
            delta_angle = delta_rotation
        # print(gt, gr, gc, delta_curvature, delta_rotation)
        self.p_direction = norm(self.p_direction + delta_curvature + delta_rotation)
        # self.p_direction = norm(self.p_direction + delta_angle)
        delta_dis = VELOCITY / gt
        tmp_x = self.x_physical + torch.cos(self.p_direction) * delta_dis
        tmp_y = self.y_physical + torch.sin(self.p_direction) * delta_dis
        if outbound(tmp_x, tmp_y):
            self.p_direction = norm(self.p_direction + PI)
            return False
        else:
            self.x_physical = tmp_x
            self.y_physical = tmp_y
            return True


    """
    when in eval mode, initialize the user's postion
    """
    def init_eval_state(self, ind, evalType=0, physical_pos=np.array([0.0,0.0])):
        # print(ind)
        if evalType == 2:
            m = (int(ind / 30)) % 4
            n = (ind % 30)/30.0
            # m = (int(ind / 10)) % 4
            # n = (ind % 10)/10.0
            # n = np.random.random()
            if m == 0:
                self.x_physical = 0
                self.y_physical = HEIGHT * n
                self.p_direction = 0
            elif m == 1:
                self.x_physical = WIDTH * n
                self.y_physical = HEIGHT
                self.p_direction = -PI/2
            elif m == 2:
                self.x_physical = WIDTH
                self.y_physical = HEIGHT * n
                self.p_direction = -PI
            elif m == 3:
                self.x_physical = WIDTH * n
                self.y_physical = 0
                self.p_direction = PI / 2
        if evalType == 0 or evalType == 1:
            ratio = max(WIDTH, HEIGHT) / max(WIDTH_ALL, HEIGHT_ALL)
            if abs(self.x_virtual) < 0.1:
                self.x_physical = 0
                self.y_physical = self.y_virtual * ratio
                self.p_direction = 0
            elif abs(self.y_virtual - HEIGHT_ALL) < 0.1:
                self.x_physical = self.x_virtual * ratio
                self.y_physical = HEIGHT
                self.p_direction = -PI/2
            elif abs(self.x_virtual - WIDTH_ALL) < 0.1:
                self.x_physical = WIDTH
                self.y_physical = self.y_virtual * ratio
                self.p_direction = -PI
            elif abs(self.y_virtual) < 0.1:
                self.x_physical = self.x_virtual * ratio
                self.y_physical = 0
                self.p_direction = PI / 2
            if evalType == 0:
                self.p_direction = self.v_direction
        elif evalType == 3:
            m = (int(ind / 25)) % 4
            n = (ind % 25)/25.0
            print(m, n)
            # n = np.random.random()
            if m == 0:
                self.x_physical = 0
                self.y_physical = HEIGHT * n
                self.p_direction = 0
            elif m == 1:
                self.x_physical = WIDTH * n
                self.y_physical = HEIGHT
                self.p_direction = -PI / 2
            elif m == 2:
                self.x_physical = WIDTH
                self.y_physical = HEIGHT * n
                self.p_direction = -PI
            elif m == 3:
                self.x_physical = WIDTH * n
                self.y_physical = 0
                self.p_direction = PI / 2
        elif evalType == 4:
            self.x_physical, self.y_physical = physical_pos
        self.p_direction = self.compute_dir_to_obj()
            # self.print()
            # print(self.p_direction)

    def compute_dir_to_obj(self):
        vec = np.array([WIDTH/2-self.x_physical, HEIGHT/2-self.y_physical])
        if vec[0]==0:
            if vec[1] > 0:
                theta = PI/2
            else:
                theta = -PI/2
        else:
            tan = vec[1] / vec[0]
            theta = np.arctan(tan)
            if vec[0] < 0 and tan >= 0:
                theta = norm(theta + PI)
            elif tan < 0 and vec[0] < 0:
                theta = norm(theta + PI)
        return theta

    def step_specific_path(self, actor_critic, ind, ep=None, evalType=1, physical_pos=np.array([0.0,0.0])):
        # RunningState = RunningStats()
        collide = 0
        std1 = []
        std2 = []
        std3 = []
        x_l = []
        y_l = []
        gt_l = []
        gr_l = []
        gc_l = []
        self.v_path = self.pas_path_file[ind]
        self.v_step_pointer = 0
        if evalType == 3:
            self.v_path = self.pas_path_file[4]
        if evalType == 4:
            self.v_step_pointer = int(len(self.v_path)/2 / (ind % 10+1))

        self.x_virtual, self.y_virtual, self.v_direction, self.delta_direction_per_iter = self.v_path[
            self.v_step_pointer]
        self.init_eval_state(ind, evalType, physical_pos)
        # self.print()
        # self.obj_x, self.obj_y, _, _ = self.v_path[-1]
        # self.obj_d = norm(self.obj_d + PI)
        # self.x_physical = 3.0
        # self.y_physical = 0.0
        # self.p_direction = PI / 2
        init_obs = []
        init_obs.extend(10 * self.get_obs())
        self.current_obs = torch.Tensor(init_obs)

        i = self.v_step_pointer
        tmp_r = 0
        while i < len(self.v_path):
            # print(i)
            with torch.no_grad():
                values, action_mean, action_log_std = actor_critic.act(
                    torch.Tensor(self.current_obs).unsqueeze(0))
                dist = FixedNormal(action_mean, action_log_std)
                std1.append(action_log_std[0][0].item())
                std2.append(action_log_std[0][1].item())
                std3.append(action_log_std[0][2].item())
                # action = action_mean
                action = dist.mode()
                # action = action.clamp(-1.0, 1.0)
                # print("mean:", action_mean, '\tstd:', action_log_std, "\taction:", action)
            gt, gr, gc = split(action)
            # self.print()
            # print(gt.item(), gr.item(), gc.item())
            gt_l.append(gt.item())
            gr_l.append(gr.item())
            gc_l.append(gc.item())
            for m in range(10):
                if i > len(self.v_path) - 1:
                    signal = False
                    # self.reward += (1 - 0.1 * distance(self.x_physical, self.y_physical, self.obj_x_p, self.obj_y_p))
                    # self.reward += self.get_reward()
                    self.reward += self.final_reward()
                    break
                signal = self.physical_step(gt, gr, gc)
                self.vPathUpdate()
                x_l.append(self.x_physical)
                y_l.append(self.y_physical)
                i += 1
                if not signal:
                    self.reward -= PENALTY
                    collide += 1
                    # print("srl_id:",ind,"penalty:",PENALTY)
                    # self.reward += self.get_reward()
                    break
                elif m == 0:
                    self.reward += self.get_reward()
            # self.reward += tmp_r
                # self.reward += tmp_r
            if not signal:
                break
            # obs = self.current_obs[OBSERVATION_SPACE:]
            # obs.extend(self.get_obs())
            init_obs = init_obs[OBSERVATION_SPACE:]
            init_obs.extend(self.get_obs())

            self.current_obs = torch.Tensor(init_obs)
            if OBS_NORM:
                RunningState.push(self.current_obs)
                self.current_obs = (self.current_obs - RunningState.mean()) / (torch.sqrt(RunningState.variance()) + 1e-5)
        vx_l = self.v_path[:, 0]
        vy_l = self.v_path[:, 1]
        # self.reward = self.get_reward()
        final_dis = math.sqrt((self.x_physical - self.obj_x_p) * (self.x_physical  - self.obj_x_p) +
                              (self.y_physical - self.obj_y_p) * (self.y_physical  - self.obj_y_p))
        # print(self.reward)
        return self.reward, final_dis, self.err_angle(), gt_l, gr_l, gc_l, x_l, y_l, vx_l, vy_l, std1, std2, std3, collide

    def err_angle(self):
        # print(self.path_cnt, self.p_direction.item(), self.v_direction, (self.p_direction - self.v_direction).item(),
        #       (delta_angle_norm(self.p_direction-self.v_direction).item()))
        return abs(self.get_reward_angle())*PI
        # return abs(delta_angle_norm(self.p_direction - self.obj_d_p))

    def final_reward(self):
        r = (1 - 0.05 * distance(self.x_physical, self.y_physical, self.obj_x_p, self.obj_y_p) - 0.5*self.err_angle()/PI)
        # print(0.1*distance(self.x_physical, self.y_physical, self.obj_x_p, self.obj_y_p), self.err_angle()/PI)
        return r
        # return 0
    def step_specific_path_nosrl(self, ind, evalType=0, physical_pos=np.array([0.0,0.0])):
        x_l = []
        y_l = []
        collide = 0
        self.v_path = self.pas_path_file[ind]
        self.v_step_pointer = 0
        if evalType==4:
            self.v_step_pointer = int(len(self.v_path)/2 / (ind % 10+1))
        if evalType==3:
            self.v_path = self.pas_path_file[4]
        self.x_virtual, self.y_virtual, self.v_direction, self.delta_direction_per_iter = self.v_path[
            self.v_step_pointer]
        self.init_eval_state(ind, evalType, physical_pos)
        # self.print()
        init_obs = []
        init_obs.extend(10 * self.get_obs())
        self.current_obs = init_obs
        i = self.v_step_pointer
        while i < len(self.v_path):
            gr, gt, gc = torch.Tensor([1.0]), torch.Tensor([1.0]), torch.Tensor([0.0])
            for m in range(10):
                if i > len(self.v_path) - 1:
                    signal = False
                    # self.reward -= 1
                    # self.reward += self.get_reward()
                    self.reward += self.final_reward()
                    break
                signal = self.physical_step(gt, gr, gc)
                x_l.append(self.x_physical)
                y_l.append(self.y_physical)
                self.vPathUpdate()
                i += 1
                if not signal:
                    self.reward -= PENALTY
                    collide += 1
                    # print("id:",ind,"penalty:-10")
                    # self.reward += self.get_reward()
                    break
                elif m == 0:
                #     #tmp_r = 0
                #     tmp_r = self.get_reward()
                    self.reward += self.get_reward()
            # self.reward += tmp_r
            if not signal:
                break
            obs = self.current_obs[OBSERVATION_SPACE:]
            obs.extend(self.get_obs())
            self.current_obs = obs
            obs = toTensor(obs)
        # plt.axis([0, 5.79, 0, 5.79])
        # plt.scatter(x_l, y_l, s=1, c='b')
        # plt.savefig('./plot_result/none/none_' + str(ind) + '.png')
        # plt.clf()
        #self.reward = self.get_reward()
        return self.reward, math.sqrt(
            (self.x_physical  - self.obj_x_p) * (self.x_physical - self.obj_x_p) +
            (self.y_physical  - self.obj_y_p) * (self.y_physical - self.obj_y_p)), self.err_angle(), x_l, y_l, collide

    def get_reward(self):
        # d_wall = min(self.x_physical/WIDTH, (WIDTH-self.x_physical)/WIDTH, (self.y_physical)/HEIGHT, (HEIGHT-self.y_physical)/HEIGHT)
        # r2, d1, d2, max_d = self.get_reward_distance()
        # r2 = 0
        # r1 = self.get_reward_wall() * np.exp(-5*d_wall)
        # r1 = 0
        # print(self.x_physical, self.y_physical, d_wall, np.exp(-5*d_wall))
        # r1 = self.get_reward_wall()
        # r3 = self.get_reward_angle() * (1-np.exp(-3*d2))
        # r3_f = 0
        r3 = self.get_reward_angle()
        # r3 = self.err_angle()/PI
        # print(r1, r2, r3)
        # if r3 > 0.2:
        #     r3_f = 1
        # r3_f = r3_f * (1 - np.exp(-d2))
        # r3 = 0
        # print(r1,r2,r3)
        # print(max_d, np.exp(5*(max_d-0.5)), r1, r2.item(), r3)
        # r3 = self.get_reward_angle() * np.exp(-5 * d2)
        # print(r2)
        # print(0.1 * r1, 5 * r2, r3)
        # print(r2.item(), r3)
        # if(math.isnan(r3)):
        #     self.print()
        #     exit()
        # if d2 > 0.5:
        #     r3 = 0
        # print(0.1 * r1, 5 * r2, 0.1 * r1 - 5 * r2)
        # print(d2, r3)
        # print(10 * r2, r3)

        # div =
        # torch.clamp(toTensor(1/(d_v+1e-5)), 0, 3)
        # print(5 - r2)
        # print(5 - 20 * r2)
        # print(r2)
        # print(r2, 0.5 * r3,  5- r2-0.5 * r3)
        # print((1 - r2 - 0.2 * r3)/10)
        # print(r2, 0.2 * r3, 1 - r2 - 0.2 * r3)
        # print(r2, 5* r2 , r3)
        # print(r3)
        # print(r2, 0.2 * r3, 1 - r2 - 0.2 * r3)
        # print(r2, 1 - r2)
        # print( - r2)
        # print(d2, r3.item())
        # print(5 * r2, r3, 1-5*r2-r3, 5 * r2 - r3)
        # print(r2, r3, 1 - r2 -r3, r2- r3)
        # self.r1.append(r1)
        # self.r2.append(r2)
        # self.r3.append(r3)
        # return (1 - 0.1 * r1 - 5 * r2 - r3)/10
        # print(r3)
        # print(0.4*r1, 10 * r2, 3*r3,  (0.4*r1- 10 *r2 - 3*r3))
        # print(( 0.4*r1- 10 *r2 - 3*r3)/10)
        # print(3*r2, r3)
        # return torch.Tensor([0.0])
        # print(d2, 0.2 * r1 , 5*r2, r3)
        # print(d2, r1, r2, r3, 0.2*r1 - r2-r3)
        # self.r_l.append(r3)
        # self.distance.append(d1)
        # if len(self.r_l)==500:
        #     plt.scatter(np.array(self.distance), np.array(self.r_l))
        #     plt.show()
        #     plt.cla()
        #     print(np.array(self.r_l).mean())
        # return ( 0.05 * r1 - r2 - 2*r3)/100
        # return (0.5*r1 - r2_f - r3_f)/100
        # return (0.5 + 0.5*r1 -  5*r2 -2* r3)/100
        return (0.5 - r3)/10
        # return (0.5 +0.5*r1 -2*r2)/100
        # return (1.0 + 0.2 * r1 - 0.5 * r2) / 100
        # return r1/100
        # return torch.Tensor([0.0])

    def print(self):
        print("physical:", self.x_physical, " ", self.y_physical, " ", self.p_direction)
        print("virtual:", self.x_virtual, " ", self.y_virtual, " ", self.v_direction)

    def get_reward_distance(self):
        # d1 = distance((self.x_physical + DELTA_X), (self.y_physical + DELTA_Y),
        #               self.obj_x, self.obj_y)
        # d1 = distance((self.x_physical + DELTA_X)/WIDTH_ALL, (self.y_physical + DELTA_Y)/WIDTH_ALL,
        #               self.obj_x/WIDTH_ALL, self.obj_y/WIDTH_ALL)
        # d2 = distance(self.x_virtual/WIDTH_ALL, self.y_virtual/WIDTH_ALL, self.obj_x/WIDTH_ALL, self.obj_y/WIDTH_ALL)
        # d1 = distance(self.x_physical,self.y_physical,self.obj_x_p, self.obj_y_p)
        d1_ratio = distance((self.x_physical+DELTA_X)/WIDTH_ALL, (self.y_physical+DELTA_Y)/HEIGHT_ALL, (self.obj_x_p+DELTA_X)/WIDTH_ALL, (self.obj_y_p+DELTA_Y)/HEIGHT_ALL)
        # d2 = distance(self.x_virtual, self.y_virtual, self.obj_x, self.obj_y)
        d2_ratio = distance(self.x_virtual/WIDTH_ALL, self.y_virtual/HEIGHT_ALL, self.obj_x/WIDTH_ALL, self.obj_y/WIDTH_ALL)
        # delta_distance_pow = abs((d1 - d2) * (d1 - d2))
        # print("locate:",  self.x_physical, self.y_physical, self.x_virtual, self.y_virtual )
        # delta_distance = abs((d1 - d2))
        delta_distance_ratio = abs(d1_ratio-d2_ratio) * np.exp(-3*d2_ratio)
        # delta_distance_ratio = abs(d1_ratio-d2_ratio)
        # delta_distance_ratio = abs(d1 - d2)
        # print(d2_ratio, d1_ratio)
        # print(d2, delta_distance_pow, delta_distance_pow/(d2+1))
        # print(d1, d2, delta_distance_pow)
        # delta_distance_ratio *= np.sqrt(2) * np.exp(-5 * d2_ratio)
        # print(d2_ratio)
        max_d = max(abs(self.x_physical-self.obj_x_p)/WIDTH, abs(self.y_physical-self.obj_y_p)/HEIGHT)
        return toTensor(delta_distance_ratio), d1_ratio, d2_ratio, max_d
        # return toTensor(delta_distance_pow / (1+d2) / 4.0941)
    def get_reward_wall(self):
        x, y = self.x_physical / WIDTH, self.y_physical / HEIGHT
        # print(x,y)
        cos, sin = torch.cos(self.p_direction), torch.sin(self.p_direction)
        if sin == 0 and cos > 0:
            r = 1 - x + min(y, 1 - y)
        elif sin == 0 and cos < 0:
            r = x + min(y, 1 - y)
        elif cos == 0 and sin > 0:
            r = 1 - y + min(x, 1 - x)
        elif cos == 0 and sin < 0:
            r = y + min(x, 1 - x)
        else:
            a = sin / cos
            b = y - a * x
            min_len1 = min_length_direction(x, y, a, b, cos)  # ray from the walking direction
            a_ = -1 / a
            b_ = y + 1 / a * x
            min_len2 = min_length(x, y, a_, b_)  # ray tangent to the walking direction
            r = min_len1 + min_len2
        # return distance(x, y, 0.5, 0.5) * 1.4
        return r / np.sqrt(2)

    '''
    This method compute the angle error between the person and the target
    '''
    def get_reward_angle(self):
        # return self.err_angle() / (PI)
        # vec1 = np.array([self.obj_x_p-self.x_physical, self.obj_y_p -self.y_physical])
        # vec2 = np.array([self.obj_x - self.x_virtual, self.obj_y - self.y_virtual])

        vec1 = np.array([np.cos(self.obj_d_p), np.sin(self.obj_d_p)])
        vec2 = np.array([np.cos(self.obj_d),   np.sin(self.obj_d)])

        vec3 = np.array([np.cos(self.p_direction), np.sin(self.p_direction)])
        vec4 = np.array([np.cos(self.v_direction), np.sin(self.v_direction)])
        vec1 = normaliztion(vec1)
        vec2 = normaliztion(vec2)
        ang1 = np.arccos(np.clip(np.dot(vec1, vec3), -1.0, 1.0))
        ang2 = np.arccos(np.clip(np.dot(vec2, vec4), -1.0, 1.0))
        # num1 = np.dot(vec1, vec3)
        # num2 = np.dot(vec2, vec4)
        if np.cross(vec1, vec3) * np.cross(vec2, vec4) < 0:
            ang = delta_angle_norm(ang1 + ang2)
        else:
            ang = delta_angle_norm(ang1 - ang2)
        # if math.isnan(ang):
        #     self.print()
        #     print(np.dot(vec1, vec3))
        #     print(np.dot(vec2, vec4))
        # print(ang1, ang2)
        return abs(ang)/PI



    '''
    scale the angle into 0-pi
    '''
def delta_angle_norm(x):
    if x >= PI:
        x = 2 * PI - x
    elif x <= -PI:
        x = x + 2 * PI
    return x

def normaliztion(x):
    if x[0]*x[0] + x[1] * x[1] == 0:
        return x
    return x /np.sqrt(x[0]*x[0] + x[1] * x[1])

def toTensor(x):
    return torch.Tensor([x])

def toPI(x):
    return x * PI / 180.0


class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else torch.Tensor([0.0])
