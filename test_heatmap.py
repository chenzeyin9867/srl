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
from tqdm import trange
sns.set()
from a2c_ppo_acktr.envs_general import *
from a2c_ppo_acktr.arguments import get_args
# from a2c_ppo_acktr.myutils import get_render_func, get_vec_normalize

sys.path.append('a2c_ppo_acktr')
# parser = argparse.ArgumentParser(description='RL')
args = get_args()

env = PassiveHapticsEnv(args.gamma, args.stack_frame, eval=True)

# We need to use the same statistics for normalization as used in training
obs = env.reset()
if args.load_epoch != 0:
    actor_critic = torch.load('./trained_models/' + args.env_name + '/%d.pth' % args.load_epoch)
print("Loading the " + args.env_name + '/_%d.pt' % args.load_epoch + ' to train')

gt = []
gr = []
gc = []
no_srl_reward = 0
# for _ in range(len(v_path)):
T = 90
cnt = 0
interval = 50


  
gt = []
gr = []
gc = []
cnt = 0
for _ in trange(1, T):
    cnt += 1
    x = WIDTH * _/T
    y = 0.0
    direction = PI/2.0
    env.reset()
    env.set(x, y, direction)  # set the initial state of +the env
    t, r, c = env.test_heatmap(actor_critic)
    # print(r)
    # print(len(ret_x), len(ret_y), len(t), len(r), len(c))
    gt.extend(t)
    gr.extend(r)
    gc.extend(c)

t_np = np.array(gt).reshape((cnt, -1)).transpose(1, 0)
r_np = np.array(gr).reshape((cnt, -1)).transpose(1, 0)
c_np = np.array(gc).reshape((cnt, -1)).transpose(1, 0)
# t_np = np.resize(t_np, (30, 30))
# r_np = np.resize(r_np, (30, 30))
# c_np = np.resize(c_np, (30, 30))
    # else:
    #     t_np += np.array(gt).reshape((cnt, -1)).transpose(1, 0)
    #     r_np += np.array(gr).reshape((cnt, -1)).transpose(1, 0)
    #     c_np += np.array(gc).reshape((cnt, -1)).transpose(1, 0)
sns_plot = sns.heatmap(t_np, cmap="rainbow")
# save_dir = 'plot_result/heat_map/' + args.load_param.split('/')[-1]
save_dir = os.path.join('plot_result/heat_map', args.env_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.savefig(save_dir + '/gt_heatmap.png')
plt.clf()

sns_plot = sns.heatmap(r_np, cmap="rainbow")
plt.title('Gr')
# fig.savefig("heatmap.pdf", bbox_inches='tight') # 减少边缘空白
plt.savefig(save_dir + '/gr_heatmap.png')
plt.clf()

sns_plot = sns.heatmap(c_np, cmap="rainbow")
plt.title('Gc')
# fig.savefig("heatmap.pdf", bbox_inches='tight') # 减少边缘空白
plt.savefig(save_dir + '/gc_heatmap.png')
plt.clf()