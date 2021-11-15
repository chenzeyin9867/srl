import numpy as np
import torch
import os
from gym.spaces.box import Box
from srl_core.distributions import FixedNormal
import random
import math

VELOCITY = 1.4 / 60.0
HEIGHT, WIDTH = 20.0 , 20.0
PI = np.pi
PENALTY = torch.Tensor([1.0])
OBSERVATION_SPACE = 3

class PassiveHapticsEnv(object):
    def __init__(self, gamma, num_frame_stack, path, random=False, eval=False):
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

        if not eval:
            self.path_file = np.load(os.path.join(path, 'train.npy'), allow_pickle=True)
        else:
            self.path_file = np.load(os.path.join(path, 'eval.npy'),  allow_pickle=True)
        self.v_path = self.path_file[self.path_cnt]
        self.v_step_pointer = 0         # the v_path counter


    def get_obs(self):
        """
        scale the physical and virtual space into the same square box inside [-1,1]
        """
        state = [ self.x / WIDTH, 
                  self.y / HEIGHT,
                  (self.o + PI) / (2.0 * PI),
                #   self.delta_direction_per_iter,
                  ]
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
        r1 = self.get_reward_wall()
        return r1


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
    Some utils 
'''

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
    p2 = torch.Tensor([1, a + b])
    p3 = torch.Tensor([-b / a, 0])
    p4 = torch.Tensor([(1 - b) / a, 1.])
    p = torch.cat((p1, p2, p3, p4))
    p = p.reshape((4, 2))
    p = p[p[:, 0].argsort(), :]
    if cos > 0:
        c, d = p[2]
    else:
        c, d = p[1]
    len = distance(x, y, c, d)
    return len


def min_length(x, y, a, b):  # min length of the line y = ax+b with intersection with the bounding box of [0,1]
    p1 = torch.Tensor([0, b])
    p2 = torch.Tensor([1, a + b])
    p3 = torch.Tensor([-b / a, 0])
    p4 = torch.Tensor([(1 - b) / a, 1.])
    p = torch.cat((p1, p2, p3, p4))
    p = p.reshape((4, 2))
    p = p[p[:, 0].argsort(), :]
    c, d = p[1]
    e, f = p[2]
    return min(distance(x, y, c, d), distance(x, y, e, f))


def distance(x, y, a, b):
    return math.sqrt((x - a) * (x - a) + (y - b) * (y - b))


def toTensor(x):
    return torch.Tensor(x)