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
from srl_core.envs_general import *
from srl_core.arguments import get_args
from srl_core.model import *
# from srl_core.myutils import get_render_func, get_vec_normalize

sys.path.append('srl_core')
# parser = argparse.ArgumentParser(description='RL')
args = get_args()

env = PassiveHapticsEnv(args.gamma, args.stack_frame, args.data, eval=True)

# We need to use the same statistics for normalization as used in training
obs = env.reset()
if args.load_epoch != 0:
    
    actor_critic = torch.load('./trained_models/' + args.env_name + '/%d.pth' % args.load_epoch)
print("Loading the " + args.env_name + '/%d.pt' % args.load_epoch)

gt = []
gr = []
gc = []
no_srl_reward = 0
# for _ in range(len(v_path)):
T = 30
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