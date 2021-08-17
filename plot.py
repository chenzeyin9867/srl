import numpy as np
import torch
from a2c_ppo_acktr import myutils
import argparse
from a2c_ppo_acktr.envs_general import *


# def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
#              device):
#     eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
#                               None, eval_log_dir, device, True)
#
#     vec_norm = myutils.get_vec_normalize(eval_envs)
#     if vec_norm is not None:
#         vec_norm.eval()
#         vec_norm.ob_rms = ob_rms
#
#     eval_episode_rewards = []
#
#     obs = eval_envs.reset()
#     eval_recurrent_hidden_states = torch.zeros(
#         num_processes, actor_critic.recurrent_hidden_state_size, device=device)
#     eval_masks = torch.zeros(num_processes, 1, device=device)
#
#     while len(eval_episode_rewards) < 10:
#         with torch.no_grad():
#             _, action, _, eval_recurrent_hidden_states = actor_critic.act(
#                 obs,
#                 eval_recurrent_hidden_states,
#                 eval_masks,
#                 deterministic=True)
#
#         # Obser reward and next obs
#         obs, _, done, infos = eval_envs.step(action)
#
#         eval_masks = torch.tensor(
#             [[0.0] if done_ else [1.0] for done_ in done],
#             dtype=torch.float32,
#             device=device)
#
#         for info in infos:
#             if 'episode' in info.keys():
#                 eval_episode_rewards.append(info['episode']['r'])
#
#     eval_envs.close()
#
#     print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
#         len(eval_episode_rewards), np.mean(eval_episode_rewards)))


def rdw_evaluate(actor_critic, seed, num_processes, gamma, log_dir, device, stack_frame_num, ep, flag):
    v_path = np.load('./Dataset/eval_path.npy', allow_pickle=True)
    env = make_rdw_env(seed, num_processes, gamma, log_dir, device, stack_frame_num)
    # env.reset()
    eval_path = v_path[:]
    reward = 0
    r_none = 0
    for t in range(0, 50):
        env.reset()
        ret, _, _, _ = env.step_specific_path(actor_critic, eval_path[t], t, ep)
        reward += ret
        if flag == 0:
            env.reset()
            r_none += env.step_specific_path_nosrl(eval_path[t], t)
    flag = 1
    return reward, r_none, flag

def PassiveHapticRdwEvaluate(actor_critic, seed, num_processes, gamma, log_dir, device, stack_frame_num, ep, flag,
                             env_name, random=False, num=50, draw=True, evalType=1):
    env = make_passive_haptics_env(seed, num_processes, gamma, log_dir, device, stack_frame_num, random, eval=True)
    reward = 0
    r_none = 0
    distance_physical = 0
    dis_nosrl = 0
    angle_srl = 0
    angle_none = 0
    ret_srl_list = []
    ret_none_list = []
    std_list1 = []
    std_list2 = []
    std_list3 = []
    gt_list = []
    gr_list = []
    gc_list = []
    collide = 0
    collide_ = 0
    t = 0
    env.reset()
    ret, dis, angle, gt, gr, gc, x, y, vx, vy, std1, std2, std3, c = env.step_specific_path(actor_critic, t, ep, evalType)
    ret_srl_list.append(dis)
    collide += c
    std_list1.extend(std1)
    std_list2.extend(std2)
    std_list3.extend(std3)
    gt_list.extend(gt)
    gr_list.extend(gr)
    gc_list.extend(gc)
    reward += ret
    angle_srl += angle
    distance_physical += dis
    #if flag == 0:
    env.reset()
    r_, dis_, angle_,  x_, y_, c_= env.step_specific_path_nosrl(t, evalType)
    ret_none_list.append(dis_)
    collide_ += c_
    r_none += r_
    dis_nosrl += dis_
    angle_none += angle_

    plt.figure(1, figsize=(15, 5))
    p_ax = plt.subplot(1, 3, 2)
    v_ax = plt.subplot(1, 3, 1)
    pr_ax = plt.subplot(1, 3, 3)
    p_ax.scatter(WIDTH / 2.0, HEIGHT / 2.0, s=10)
    pr_ax.scatter(WIDTH / 2.0, HEIGHT / 2.0, s=10)
    v_ax.scatter(WIDTH_ALL / 2.0, HEIGHT_ALL / 2.0, s=10)
    for t in range(len(vx)):
        v_ax = plt.subplot(1, 3, 1)
        p_ax = plt.subplot(1, 3, 2)
        pr_ax = plt.subplot(1, 3, 3)
        v_ax.axis([0.0, WIDTH_ALL, 0.0, HEIGHT_ALL])
        p_ax.axis([0.0, WIDTH, 0.0, HEIGHT])
        pr_ax.axis([0., WIDTH, 0., HEIGHT])
        if t < len(x):
            p_ax.scatter(x[t], y[t], s=0.2, c='r')
        if t < len(vx):
            v_ax.scatter(vx[t], vy[t], s=0.2, c='r')
        if t < len(x_):
            pr_ax.scatter(x_[t], y_[t], s=0.2, c='r')
        if t < min(len(x), len(x_), len(vx))  and t % 10 == 0:
            k = int(t/10)
            print("gt:{:.2f}\tgr:{:.2f}\tgc:{:.2f}".format(gt[k], gr[k], gc[k]))
        plt.pause(0.01)
    plt.show()



if __name__ == '__main__':
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
    param_name = os.path.join(args.load_param)
    print('loading the ————', param_name)

    actor_critic = torch.load(param_name)
    num = 1
    draw = False
    print('running ', num, " paths:")

    PassiveHapticRdwEvaluate(actor_critic, args.seed,
                                         1, 0.99, None, None, 10, None, 0, args.env_name, False, num, draw, evalType=2)


