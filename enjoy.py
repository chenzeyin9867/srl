import argparse
import os
# workaround to unpickle olf model files
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

from srl_core.envs import make_rdw_env, make_passive_haptics_env

# from srl_core.myutils import get_render_func, get_vec_normalize

sys.path.append('srl_core')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--load_param',
    default='trained_models/ppo/18_rdw_731.pt',
    type=str,
    help='the pth file you need to load'
)
args = parser.parse_args()

args.det = not args.non_det

# env = make_rdw_env(1, 1, 0.99, None, None, 10)
env = make_passive_haptics_env(1, 1, 0.99, None, None, 10)
# We need to use the same statistics for normalization as used in training
masks = torch.zeros(1, 1)

obs = env.reset()
v_path = np.load('./Dataset/new_eval_path.npy', allow_pickle=True)
eval_path = v_path[:]
interval = 250
file_name = os.listdir(args.load_param)
len = len(file_name)
# for _ in range(len(v_path)):
for iter in range(int(len / interval * 5)):
    ind = iter * interval
    param_name = os.path.join(args.load_param, str(ind) + '.pth')
    print('loading the ————', param_name)
    actor_critic = torch.load(param_name)
    ret = 0
    reward = 0
    gt = []
    gr = []
    gc = []
    no_srl_reward = 0
    for _ in range(2000):
        if _ % 100 == 0:
            print("iter:%d/ep:%d" % (iter, _))
        env.reset()
        ret, t, r, c = env.step_specific_path(actor_critic, eval_path[_], _, ep=None)
        env.reset()
        no_srl_reward += env.step_specific_path_nosrl(eval_path[_], _)
        reward += ret
        gt.extend(t)
        gr.extend(r)
        gc.extend(c)
    print("param_name", param_name, "\tSRL:", reward, "\t|No_SRL:", no_srl_reward,
          "accelerate:{:.4f}%".format(((reward - no_srl_reward) / no_srl_reward * 100).item()))

    #  plot the hist statistic
    plt.hist(gt, bins=50)
    plt.yscale('log')
    plt.grid(True)
    save_dir = './plot_result/hist/' + args.load_param.split('/')[1] + str(ind)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir + '/Gt_hist.png')
    plt.clf()
    plt.hist(gr, bins=50)
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(save_dir + '/Gr_hist.png')
    plt.clf()
    plt.hist(gc, bins=50)
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(save_dir + '/Gc_hist.png')
    plt.clf()
 