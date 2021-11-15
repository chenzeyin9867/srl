import glob
import os

import torch
import torch.nn as nn

# from srl_core.envs import VecNormalize


# Get a render function
# def get_render_func(venv):
#     if hasattr(venv, 'envs'):
#         return venv.envs[0].render
#     elif hasattr(venv, 'venv'):
#         return get_render_func(venv.venv)
#     elif hasattr(venv, 'env'):
#         return get_render_func(venv.env)
#
#     return None


# def get_vec_normalize(venv):
#     if isinstance(venv, VecNormalize):
#         return venv
#     elif hasattr(venv, 'venv'):
#         return get_vec_normalize(venv.venv)
#
#     return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
        # self._bias = nn.Parameter(torch.Tensor[])

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

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