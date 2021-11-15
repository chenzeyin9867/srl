import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from srl_core.envs_general import *
from srl_core.arguments import get_args

def srlEvaluate(actor_critic, ep, flag, **kwargs):
    env = PassiveHapticsEnv(kwargs['gamma'],  kwargs['stack_frame'], eval=True)
    reward      = 0
    r_none      = 0
    collide     = 0
    collide_    = 0
    srl_length  = 0.0
    base_length = 0.0
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
        ret, gt, gr, gc, x, y, std1, std2, std3, c, length = env.step_specific_path(actor_critic, t, evalType)
        collide += c
        srl_length += length
        std_list1.extend(std1)
        std_list2.extend(std2)
        std_list3.extend(std3)
        gt_list.extend(gt)
        gr_list.extend(gr)
        gc_list.extend(gc)
        reward += ret
        # if flag == 0:
        env.reset()
        r_,  x_, y_, c_, length_ = env.step_specific_path_nosrl(t, evalType)
        base_length += length_
        collide_ += c_
        r_none += r_
        
        if draw:
            plt.figure(1, figsize=(10, 5))
            plt_srl = plt.subplot(1, 2, 2)
            plt_none = plt.subplot(1, 2, 1)
            # plt_noSRL = plt.subplot(1, 3, 3)
            plt_none.set_title('None')
            plt_srl.set_title('SRL')
            # plt_noSRL.set_title('noSRL')

            plt_srl.axis('scaled')
            plt_srl.axis([0.0, WIDTH, 0.0, HEIGHT])
            plt_none.axis('scaled')
            plt_none.axis([0.0, WIDTH, 0.0, HEIGHT])

            plt_srl.scatter(np.array(x), np.array(y), label='SRL', s=1, c='r')
            plt_none.scatter(np.array(x_), np.array(y_), label='NONE', s=1, c='g')

            # plt_noSRL.scatter(WIDTH / 2.0, HEIGHT / 2.0, s=10)
            name = {1: "same_edge", 2: "random", 3: "shuffle"}
            if ep is None:
                if not os.path.exists('./plot_result/general'):
                    os.makedirs('./plot_result/general')
                plt.savefig('./plot_result/general/' + name[evalType] + '_' + str(t) + '.png')
                plt.clf()
                plt_none.cla()
                plt_srl.cla()
            elif ep % 100 == 0:
                if not os.path.exists('./plot_result/%s/ep_%d' % (kwargs['env_name'], ep)):
                    os.makedirs('./plot_result/%s/ep_%d' % (kwargs['env_name'], ep))
                plt.savefig('./plot_result/%s/ep_%d/srl_%d.png' % (kwargs['env_name'], ep, t))
            plt.clf()
            plt_none.cla()
            plt_srl.cla()

    flag = 1

    length = srl_length / num
    length_ = base_length / num
    return reward, r_none, flag, std_list1, std_list2, std_list3, gt_list, gr_list, gc_list, collide, collide_, length, length_







if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='RL')
    args = get_args()
    if args.load_epoch != 0:
        actor_critic = torch.load('./trained_models/' + args.env_name + '/%d.pth' % args.load_epoch)
    print("Loading the " + args.env_name + '/_%d.pt' % args.load_epoch + ' to train')
    num = 5000
    draw = False
    print('running ', num, " paths:")
    params = args.__dict__
    # test_frames = params['']
    
    r_eval, r_none, flag, std1, std2, std3, gt, gr, gc, c, c_, l, l_ = srlEvaluate(actor_critic, j, flag, **params)
    len = len()
    left = int(len/4)
    right = int(len*3/4)
    print("TYPE3:")
    print("SRL:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\tIQR:{:.2f}".format(distance_physical/num, r_1[0], r_1[int(num/2)], r_1[num-1], angle_srl.item()/num,r_1[right]-r_1[left] ), "\tcollide", collide)
    print("None:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\tIQR:{:.2f}".format(dis_nosrl/num, r_2[0], r_2[int(num/2)], r_2[num-1], angle_none.item()/num, r_2[right]-r_2[left]), "\tcollide", collide_)
    print("FRANK:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\t\t\tIQR:{:.2f}".format(0.0, r_3[0], r_3[int(num/2)], r_3[num-1], r_3[right]-r_3[left]), "\tcollide", collide_frank)