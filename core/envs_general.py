import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from torch.distributions.utils import vec_to_tril_matrix
from core.distributions import FixedNormal
import random
import matplotlib.pyplot as plt
import math

VELOCITY = 1.4 / 60.0
HEIGHT, WIDTH = 20.0 , 20.0
PI = np.pi
PENALTY = torch.Tensor([1.0])
OBSERVATION_SPACE = 3



def split(action):
    a, b, c = action[0]
    a = 1.060 + 0.2   * a
    b = 1.145 + 0.345 * b
    c = 0.13 * c
    return a, b, c


def initialize(seed):
    xt, yt, dt = 0, 0, 0
    if seed == 1:
        xt = 0
        yt = random.random() * HEIGHT
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
        self.obs = []                   # current stacked infos send to the networks
        self.path_cnt = 0
        self.x = 0.0
        self.y = 0.0
        self.o = 0.0

        # if eval:
        self.path_file = np.load('/media/common/czy/srl/Dataset/Train_/train/15.npy', allow_pickle=True)
        # print("Loading the training dataset.")
        # else:
        #     self.path_file = np.load('Dataset/train/eval_path_30_angConsistence.npy', allow_pickle=True)
        #     print("Loading the eval dataset")
        self.v_path = self.path_file[self.path_cnt]
        self.v_step_pointer = 0         # the v_path counter


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
        #         # (self.obj_x_p+DELTA_X-(WIDTH_ALL)/2)/(WIDTH_ALL/2), (self.obj_y_p+DELTA_Y-(HEIGHT_ALL)/2)/(HEIGHT_ALL/2), (self.obj_d_p)/(PI),
        #         # (self.obj_x-(WIDTH_ALL/2))/(WIDTH_ALL/2), (self.obj_y-(HEIGHT_ALL/2))/(HEIGHT_ALL/2), (self.obj_d) / (PI)
        #         self.delta_direction_per_iter
        #         ]

        state = [ self.x / WIDTH, 
                  self.y / HEIGHT,
                  (self.o + PI) / (2.0 * PI),
                #   self.delta_direction_per_iter,
                  ]
        
        # print(state)
        return state 

    def reset(self):
        '''
        Reset the state when one simulation path ends.
        '''
        self.obs = []
        # Update the path
        self.v_path = self.path_file[self.path_cnt]
        self.path_cnt += 1  # next v_path
        self.path_cnt = self.path_cnt % len(self.path_file)  # using the training data iteratively

        self.v_step_pointer = 0
        self.time_step = 0
        # delta orientation next step
        self.delta_direction_per_iter = 0
        seed = random.randint(1, 4)
        self.delta_direction_per_iter = self.v_path[self.v_step_pointer]   # initialize the user in virtual space
        self.x, self.y, self.o = initialize(seed)

        # initial frame stack
        self.obs.extend(self.num_frame_stack * self.get_obs())
        # self.obs = torch.Tensor(self.obs)
        self.reward = 0.0    
        return toTensor(self.obs)

    def step(self, action):
        """
        step forward for K times using the same action based on action-repeat strategy.
        """
        gt, gr, gc = split(action)
        k = random.randint(5, 15)   # action repetition
        reward = 0.0                # initial reward for this step period
        for ep in range(k):  # for every iter, get the virtual path info, and steering
            self.vPathUpdate()
            signal = self.physical_step(gt, gr, gc)  # steering the physical env using the actions
            self.time_step += 1

            if not signal:
                reward = reward - PENALTY
                break
            if self.v_step_pointer >= len(self.v_path) - 1:
                break
                                                      
            if ep == 0:                      # only compute reward once due to the action repeat strategy
                reward = self.get_reward()

        obs = self.obs[OBSERVATION_SPACE:]   # update the observation after k steps 
        obs.extend(self.get_obs())
        self.obs = obs
        cur_obs = toTensor(obs)


        ret_reward = reward
        self.reward += reward

        if not signal:  # reset the env when leave the tracking space
            bad_mask = 1
            self.reset()
            return cur_obs, ret_reward, [1], [bad_mask]
        elif signal and self.v_step_pointer >= len(self.v_path) - 1: # successfully end one episode, get the final reward
            self.reset()
            return cur_obs, ret_reward, [1], [0]
        else:
            return cur_obs, ret_reward, [0], [0]

    def vPathUpdate(self):
        self.delta_direction_per_iter = self.v_path[self.v_step_pointer]  # unpack the next timestep virtual value
        self.v_step_pointer += 1

    def physical_step(self, gt, gr, gc):
        delta_dis = VELOCITY / gt
        delta_curvature = gc * delta_dis
        delta_rotation = self.delta_direction_per_iter / gr

        self.x = self.x + torch.cos(torch.Tensor([self.o])) * delta_dis
        self.y = self.y + torch.sin(torch.Tensor([self.o])) * delta_dis
        if outbound(self.x, self.y):
            return False
        self.o = norm(self.o + delta_curvature + delta_rotation)
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
    def init_eval_state(self, ind, evalType=0):
        if evalType == 0:
            m = (int(ind / 4)) % 4
            n = ((ind % 4) + 0.5) / 4.0
            if m == 0:
                self.x = 0
                self.y = HEIGHT * n
                self.o = 0
            elif m == 1:
                self.x = WIDTH * n
                self.y = HEIGHT
                self.o = -PI/2
            elif m == 2:
                self.x = WIDTH
                self.y= HEIGHT * n
                self.o = -PI
            elif m == 3:
                self.x = WIDTH * n
                self.y = 0
                self.o = PI / 2


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

    def step_specific_path(self, actor_critic, ind, evalType):
        collide = 0
        length = 0.0
        std1 = []
        std2 = []
        std3 = []
        x_l = []
        y_l = []
        gt_l = []
        gr_l = []
        gc_l = []
        self.v_path = self.path_file[ind]
        self.v_step_pointer = 0

        self.delta_direction_per_iter = self.v_path[self.v_step_pointer]
        self.init_eval_state(ind, evalType)

        init_obs = []
        init_obs.extend(10 * self.get_obs())
        obs = torch.Tensor(init_obs)

        i = self.v_step_pointer
        while i < len(self.v_path):
            with torch.no_grad():
                _value, action_mean, action_log_std = actor_critic.act(
                    torch.Tensor(obs).unsqueeze(0))
                dist = FixedNormal(action_mean, action_log_std)
                std1.append(action_log_std[0][0].item())
                std2.append(action_log_std[0][1].item())
                std3.append(action_log_std[0][2].item())
                action = dist.mode()
            gt, gr, gc = split(action)
            gt_l.append(gt.item())
            gr_l.append(gr.item())
            gc_l.append(gc.item())
            for m in range(10):
                if i > len(self.v_path) - 1:
                    signal = False
                    break
                signal = self.physical_step(gt, gr, gc)
                self.vPathUpdate()
                x_l.append(self.x)
                y_l.append(self.y)
                length += VELOCITY 
                i += 1
                if not signal:
                    self.reward -= PENALTY
                    collide += 1
                    break
                elif m == 0:
                    self.reward += self.get_reward()
            if not signal:
                break
            init_obs = init_obs[OBSERVATION_SPACE:]
            init_obs.extend(self.get_obs())
            obs = torch.Tensor(init_obs)

        # vx_l = self.v_path[:, 0]
        # vy_l = self.v_path[:, 1] 
                  
        return self.reward, gt_l, gr_l, gc_l, x_l, y_l, std1, std2, std3, collide, length

    def err_angle(self):

        return abs(self.get_reward_angle())*PI
        # return abs(delta_angle_norm(self.p_direction - self.obj_d_p))

    def final_reward(self):
    
        # r = (10 - 0.5 * distance(self.x_physical, self.y_physical, self.obj_x_p, self.obj_y_p) - 0.5*self.err_angle()/PI)
        # r = (1.0 - self.err_angle()/PI)
        # print(distance(self.x_physical/WIDTH, self.y_physical/HEIGHT, self.obj_x_p/WIDTH, self.obj_y_p/HEIGHT))
        r = 10 * (1 - math.pow(distance(self.x_physical/WIDTH, self.y_physical/HEIGHT, self.obj_x_p/WIDTH, self.obj_y_p/HEIGHT), 1))
        # r = 10 * (2 - math.pow(distance(self.x_physical/WIDTH, self.y_physical/HEIGHT, self.obj_x_p/WIDTH, self.obj_y_p/HEIGHT), 1) - self.err_angle()/ PI)
        # print(0.1*distance(self.x_physical, self.y_physical, self.obj_x_p, self.obj_y_p), self.err_angle()/PI)
        # return r 
        # r = torch.Tensor([0.0])
        return r

    def step_specific_path_nosrl(self, ind, evalType=1):
        x_l = []
        y_l = []
        collide = 0 
        length = 0.0
        self.v_path = self.path_file[ind]
        self.v_step_pointer = 0
        self.delta_direction_per_iter = self.v_path[self.v_step_pointer]
        self.init_eval_state(ind)

        init_obs = []
        init_obs.extend(10 * self.get_obs())
        self.current_obs = init_obs
        i = self.v_step_pointer
        assign = False
        while i < len(self.v_path):
            gr, gt, gc = torch.Tensor([1.0]), torch.Tensor([1.0]), torch.Tensor([0.0])
            for m in range(10):
                if i > len(self.v_path) - 1:
                    break
                signal = self.physical_step(gt, gr, gc)
                x_l.append(self.x)
                y_l.append(self.y)
                self.vPathUpdate()
                length += VELOCITY
                i += 1
                if not signal:
                    self.reward -= PENALTY
                    collide += 1
                    break
                elif m == 0:
                    self.reward += self.get_reward()
            if not signal:
                break
            obs = self.current_obs[OBSERVATION_SPACE:]
            obs.extend(self.get_obs())
            self.current_obs = obs
            obs = toTensor(obs)

  
        return self.reward, x_l, y_l, collide, length

    def get_reward(self):
        # d_wall = min(self.x_physical/WIDTH, (WIDTH-self.x_physical)/WIDTH, (self.y_physical)/HEIGHT, (HEIGHT-self.y_physical)/HEIGHT)
        # r2 = self.get_reward_distance()
        r1 = self.get_reward_wall()
        # r3 = self.get_reward_angle()
        return r1


    def print(self):
        print("physical:", self.x_physical, " ", self.y_physical, " ", self.p_direction)
        print("virtual:", self.x_virtual, " ", self.y_virtual, " ", self.v_direction)

    def get_reward_distance(self):
        d1_ratio = distance((self.x_physical+DELTA_X)/WIDTH_ALL, (self.y_physical+DELTA_Y)/HEIGHT_ALL, (self.obj_x_p+DELTA_X)/WIDTH_ALL, (self.obj_y_p+DELTA_Y)/HEIGHT_ALL)
        d2_ratio = distance(self.x_virtual/WIDTH_ALL, self.y_virtual/HEIGHT_ALL, self.obj_x/WIDTH_ALL, self.obj_y/WIDTH_ALL)
        # delta_distance_ratio = abs(d1_ratio-d2_ratio) * np.exp(-3*d2_ratio)
        delta_distance_ratio = abs(d1_ratio-d2_ratio)
        # max_d = max(abs(self.x_physical-self.obj_x_p)/WIDTH, abs(self.y_physical-self.obj_y_p)/HEIGHT)
        return toTensor(delta_distance_ratio)

    def get_reward_wall(self):
        '''
        reward from the boundary based on SRL
        '''
        x, y = self.x / WIDTH, self.y / HEIGHT  # normalize to [0,1]

        cos, sin = torch.cos(self.o), torch.sin(self.o)
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
            min_len1 = min_length_direction(x, y, a, b, cos)  # distance along the walking direction
            a_ = -1 / a
            b_ = y - a_ * x
            min_len2 = min_length(x, y, a_, b_)  # distance vertical to 
            r = min_len1 + min_len2
        # return distance(x, y, 0.5, 0.5) * 1.4
        return r 

    '''
    This method compute the angle error between the person and the target
    '''
    def get_reward_angle(self):
        # return self.err_angle() / (PI)
        vec1 = np.array([self.obj_x_p-self.x_physical, self.obj_y_p -self.y_physical])
        vec2 = np.array([self.obj_x - self.x_virtual, self.obj_y - self.y_virtual])

        # vec1 = np.array([np.cos(self.obj_d_p), np.sin(self.obj_d_p)])
        # vec2 = np.array([np.cos(self.obj_d),   np.sin(self.obj_d)])

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

    def set(self, x, y, d):
        self.x, self.y, self.o = x, y, d
    
    def test_heatmap(self, actor_critic):
        gt, gr, gc = [], [], []
        init_obs = []
        init_obs.extend(10 * self.get_obs())
        obs = torch.Tensor(init_obs)
        self.delta_direction_per_iter = 0
        while 1:
            with torch.no_grad():
                _value, action_mean, action_log_std = actor_critic.act(torch.Tensor(obs).unsqueeze(0))
                dist = FixedNormal(action_mean, action_log_std)
                action = dist.mode()
            t, r, c = split(action)
            gt.append(t.item())
            gr.append(r.item())
            gc.append(c.item())
            for m in range(10):
                signal = self.physical_step(1.0, 1.0, 0.0)
                if not signal:
                    break
            # self.vPathUpdate()
            if not signal:
                break

            init_obs = init_obs[OBSERVATION_SPACE:]
            init_obs.extend(self.get_obs())
            obs = torch.Tensor(init_obs)
        
        return gt, gr, gc


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
