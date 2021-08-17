import argparse
import os
# workaround to unpickle olf model files
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set()
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs, make_rdw_env
# from a2c_ppo_acktr.myutils import get_render_func, get_vec_normalize

sys.path.append('a2c_ppo_acktr')
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

env = make_rdw_env(1, 1, 0.99, None, None, 10)

# We need to use the same statistics for normalization as used in training
obs = env.reset()
v_path = np.load('./Dataset/eval_path.npy', allow_pickle=True)
eval_path = v_path[:]
reward = 0
x_l = []
y_l = []
gt = []
gr = []
gc = []
no_srl_reward = 0
# for _ in range(len(v_path)):
T = 30
cnt = 0
interval = 50
file_name = os.listdir(args.load_param)
len = len(file_name)
for iter in range(int(len / interval * 5)+1):
    ind = iter * interval
    param_name = os.path.join(args.load_param, str(ind)+'.pth')
    gt = []
    gr = []
    gc = []
    cnt = 0
    print(param_name)
    actor_critic = torch.load(param_name)
    for _ in range(1, T):
        cnt += 1
        x = 5.79 * _/T
        y = 0.0
        direction = np.pi / 2.0
        #("ep:", str(_))
        env.reset()
        env.set(x, y, direction)  # set the initial state of +the env
        ret_x, ret_y, t, r, c = env.test_heatmap(actor_critic, eval_path[_], _, ep=None)
        # print(r)
        # print(len(ret_x), len(ret_y), len(t), len(r), len(c))
        gt.extend(t)
        gr.extend(r)
        gc.extend(c)

    t_np = np.array(gt).reshape((cnt, -1)).transpose(1, 0)
    r_np = np.array(gr).reshape((cnt, -1)).transpose(1, 0)
    c_np = np.array(gc).reshape((cnt, -1)).transpose(1, 0)
    # else:
    #     t_np += np.array(gt).reshape((cnt, -1)).transpose(1, 0)
    #     r_np += np.array(gr).reshape((cnt, -1)).transpose(1, 0)
    #     c_np += np.array(gc).reshape((cnt, -1)).transpose(1, 0)
    sns_plot = sns.heatmap(t_np, cmap="rainbow")
    save_dir = 'plot_result/heat_map/' + args.load_param.split('/')[-1]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir + '/gt_heatmap_%d.png' % (iter*interval))
    plt.clf()

    sns_plot = sns.heatmap(r_np, cmap="rainbow")
    plt.title('Gr')
    # fig.savefig("heatmap.pdf", bbox_inches='tight') # 减少边缘空白
    plt.savefig(save_dir + '/gr_heatmap_%d.png' % (iter*interval))
    plt.clf()

    sns_plot = sns.heatmap(c_np, cmap="rainbow")
    plt.title('Gc')
    # fig.savefig("heatmap.pdf", bbox_inches='tight') # 减少边缘空白
    plt.savefig(save_dir + '/gc_heatmap_%d.png' % (iter*interval))
    plt.clf()