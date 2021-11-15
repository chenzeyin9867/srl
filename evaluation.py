import os
from pickle import NONE
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from srl_core.envs_general import *
from srl_core.arguments import get_args


def srlEvaluate(actor_critic, ep, **kwargs):
    env = PassiveHapticsEnv(kwargs['gamma'],  kwargs['stack_frame'], kwargs['data'], eval=True)
    reward      = 0
    collide     = 0
    length      = 0
    std_list1   = []
    std_list2   = []
    std_list3   = []
    gt_list     = []
    gr_list     = []
    gc_list     = []
    num         = kwargs['test_frames']
    draw        = kwargs['draw']
    evalType    = kwargs['path_type']
    env.reset()
    for t in trange(0, num):
        env.reset()
        if actor_critic == None:
            ret, x, y, c, l = env.step_specific_path_nosrl(t, evalType)
        else:  
            ret, gt, gr, gc, x, y, std1, std2, std3, c, l = env.step_specific_path(actor_critic, t, evalType)
            std_list1.extend(std1)
            std_list2.extend(std2)
            std_list3.extend(std3)
            gt_list.extend(gt)
            gr_list.extend(gr)
            gc_list.extend(gc)
        collide += c
        length += l
        reward += ret
        if draw and t < num / 10.0:
            plt.figure(1, figsize=(5, 5))
            title = "SRL" if actor_critic else "None"
            # plt_srl = plt.subplot(1, 2, 2)
            # plt.set_title(title)
            # plt_noSRL.set_title('noSRL')
            plt.axis('scaled')
            plt.axis([0.0, WIDTH, 0.0, HEIGHT])
            plt.scatter(np.array(x), np.array(y), s=1, c='r')

            # plt_noSRL.scatter(WIDTH / 2.0, HEIGHT / 2.0, s=10)
            name = {1: "same_edge", 2: "random", 3: "shuffle"}
            if ep is None:
                if not os.path.exists('./plot_result/general'):
                    os.makedirs('./plot_result/general')
                plt.savefig('./plot_result/general/' + name[evalType] + '_' + str(t) + '.png')
                plt.clf()
                plt.cla()
            elif ep % 100 == 0:
                if not os.path.exists('./plot_result/%s/ep_%d' % (kwargs['env_name'], ep)):
                    os.makedirs('./plot_result/%s/ep_%d' % (kwargs['env_name'], ep))
                plt.savefig('./plot_result/%s/ep_%d/%s_%d.png' % (kwargs['env_name'], ep, title, t))
            plt.clf()
            plt.cla()

    length = length / num
    reward = reward / num
    return reward, std_list1, std_list2, std_list3, gt_list, gr_list, gc_list, collide, length



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='RL')
    args = get_args()
    if args.load_epoch != 0:
        actor_critic = torch.load('./trained_models/' + args.env_name + '/%d.pth' % args.load_epoch)
    print("Loading the " + args.env_name + '/%d.pt' % args.load_epoch)
    num = args.test_frames
    draw = args.draw
    print('running ', num, " paths")
    params = args.__dict__
    # test_frames = params['']
    
    r_eval, r_none, flag, std1, std2, std3, gt, gr, gc, c, c_, l, l_ = srlEvaluate(actor_critic, None, 1, **params)
    print("******** Customized Output.\n ********")
    print("basic_reward:%.4f \tsrl_reward:%.4f" % (r_none/num, r_eval/num))
    print("basic_length:%.4f \tsrl_length:%.4f" % (l_, l))