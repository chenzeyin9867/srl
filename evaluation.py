import numpy as np
import torch
import time
from a2c_ppo_acktr import myutils
import argparse
# from a2c_ppo_acktr.envs import *
from a2c_ppo_acktr.envs_general import *
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from a2c_ppo_acktr.FrankENV import *
import shutil


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
    ret_frank_list = []
    std_list1 = []
    std_list2 = []
    std_list3 = []
    gt_list = []
    gr_list = []
    gc_list = []
    collide = 0
    collide_ = 0
    #pas_path_file = np.load('./Dataset/new_passive_haptics_path_h20w30.npy', allow_pickle=True)
    for t in range(0, num):
        if t % 100 == 0 and not ep:
            print(t)
        env.reset()
        ret, dis, angle, gt, gr, gc, x, y, vx, vy, std1, std2, std3, c = env.step_specific_path(actor_critic, t, ep,
                                                                                                evalType)
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
        # if flag == 0:
        env.reset()
        r_, dis_, angle_, x_, y_, c_ = env.step_specific_path_nosrl(t, evalType)
        ret_none_list.append(dis_)
        collide_ += c_
        r_none += r_
        dis_nosrl += dis_
        angle_none += angle_


        if draw:
            plt.figure(1, figsize=(10, 5))
            plt_srl = plt.subplot(1, 2, 2)
            plt_none = plt.subplot(1, 2, 1)
            # plt_noSRL = plt.subplot(1, 3, 3)
            plt_none.set_title('virtual')
            plt_srl.set_title('physical')
            # plt_noSRL.set_title('noSRL')

            plt_srl.axis('scaled')
            # plt_srl.set_xlim([0.0, WIDTH])
            # plt_srl.set_ylim([0.0, HEIGHT])
            plt_srl.axis([0.0, WIDTH, 0.0, HEIGHT])
            # plt_noSRL.axis([0.0, WIDTH, 0.0, HEIGHT])
            # plt_srl.axis('equal')
            # plt_none.axis('equal')

            plt_none.axis('scaled')
            plt_none.axis([0.0, WIDTH_ALL, 0.0, HEIGHT_ALL])
            # plt_none.set_xlim([0.0, WIDTH_ALL])
            # plt_none.set_ylim([0.0, HEIGHT_ALL])
            plt_srl.scatter(np.array(x), np.array(y), label='SRL', s=1, c='r')
            plt_srl.scatter(np.array(x_), np.array(y_), label='NONE', s=1, c='g')
            plt_none.scatter(np.array(vx), np.array(vy), s=1, c='b')
            plt_srl.legend()
            # plt_noSRL.scatter(np.array(x_), np.array(y_), s=1, c='r')
            plt_srl.scatter(env.obj_x_p, env.obj_y_p, s=10)

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
                if not os.path.exists('./plot_result/%s/ep_%d' % (env_name, ep)):
                    os.makedirs('./plot_result/%s/ep_%d' % (env_name, ep))
                plt.savefig('./plot_result/%s/ep_%d/srl_%d.png' % (env_name, ep, t))
            plt.clf()
            plt_none.cla()
            plt_srl.cla()
            # plt_noSRL.cla()
            # print("retL", ret, "r_ :", r_, "dis: ", dis, "dis_ :" ,dis_)
        # if draw:
        #     plt.figure(1, figsize=(15, 5))
        #     plt_srl = plt.subplot(1, 3, 2)
        #     plt_none = plt.subplot(1, 3, 1)
        #     plt_noSRL = plt.subplot(1, 3, 3)
        #     plt_none.set_title('None')
        #     plt_srl.set_title('SRL')
        #     plt_noSRL.set_title('noSRL')
        #     plt_srl.axis([0.0, WIDTH, 0.0, HEIGHT])
        #     plt_noSRL.axis([0.0, WIDTH, 0.0, HEIGHT])
        #     # plt_srl.axis('equal')
        #     # plt_none.axis('equal')
        #     plt_none.axis([0.0, WIDTH_ALL, 0.0, HEIGHT_ALL])
        #     plt_srl.scatter(np.array(x), np.array(y), s=1, c='g')
        #     plt_none.scatter(np.array(vx), np.array(vy), s=1, c='b')
        #     plt_noSRL.scatter(np.array(x_), np.array(y_), s=1, c='r')
        #     plt_srl.scatter(WIDTH / 2.0, HEIGHT / 2.0, s=10)
        #     plt_noSRL.scatter(WIDTH / 2.0, HEIGHT / 2.0, s=10)
        #     if ep is None:
        #         if not os.path.exists('./plot_result/general'):
        #             os.makedirs('./plot_result/general')
        #         plt.savefig('./plot_result/general/' + str(t) + '_' + str(evalType) + '.png')
        #         plt.clf()
        #         plt_none.cla()
        #         plt_srl.cla()
        #     elif ep % 100 == 0:
        #         if not os.path.exists('./plot_result/%s/ep_%d' % (env_name, ep)):
        #             os.makedirs('./plot_result/%s/ep_%d' % (env_name, ep))
        #         plt.savefig('./plot_result/%s/ep_%d/srl_%d.png' % (env_name, ep, t))
        #         plt.clf()
        #         plt_none.cla()
        #         plt_srl.cla()
        #         plt_noSRL.cla()
        #     # print("retL", ret, "r_ :", r_, "dis: ", dis, "dis_ :" ,dis_)

    flag = 1
    r_1 = sorted(ret_srl_list)
    r_2 = sorted(ret_none_list)
    return reward, r_none, distance_physical, dis_nosrl, angle_srl, angle_none, \
           flag, r_1, r_2, std_list1, std_list2, std_list3, gt_list, gr_list, gc_list, collide, collide_

def PassiveHapticRdwEvaluateFrank(actor_critic, seed, num_processes, gamma, log_dir, device, stack_frame_num, ep, flag,
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
    ret_frank_list = []
    std_list1 = []
    std_list2 = []
    std_list3 = []
    gt_list = []
    gr_list = []
    gc_list = []
    collide = 0
    collide_ = 0
    collide_frank = 0
    pas_path_file = np.load('./Dataset/eval_path_30.npy', allow_pickle=True)
    pos = np.loadtxt('eval_position.txt')

    dir = 'result_txt/v30_sparse'
    # if os.path.exists(dir):
    #     shutil.rmtree(dir)
    #     os.mkdir(dir)
    #     os.mkdir(dir + '/none')
    #     os.mkdir(dir + '/Ours')
    #     os.mkdir(dir + '/frank')
    #     os.mkdir(dir + '/vrst')
    #     os.mkdir(dir + '/virtual')
    # else:
    #     os.mkdir(dir)
    #     os.mkdir(dir + '/none')
    #     os.mkdir(dir + '/Ours')
    #     os.mkdir(dir + '/frank')
    #     os.mkdir(dir + '/vrst')
    #     os.mkdir(dir + '/virtual')
    start_t = time.time()
    for t in range(0, num):
        if t % 10 == 0 and not ep:
            print(t)
        if t == 75:
            print("here")
        env.reset()
        physical_pos = pos[t]
        ret, dis, angle, gt, gr, gc, x, y, vx, vy, std1, std2, std3, c = env.step_specific_path(actor_critic, t, ep, evalType, physical_pos)
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
        # if flag == 0:
        env.reset()
        r_, dis_, angle_,  x_, y_, c_= env.step_specific_path_nosrl(t, evalType, physical_pos)
        ret_none_list.append(dis_)
        collide_ += c_
        r_none += r_
        dis_nosrl += dis_
        angle_none += angle_

        frank_env = FrankEnv(pas_path_file[t], t, evalType)
        if evalType==3:
            frank_env = FrankEnv(pas_path_file[1], t, evalType)
        frank_x, frank_y, x_v_flag, y_v_flag, x_p_flag, y_p_flag, tangent_x, tangent_y, dis, ang, collide_f = \
            frank_env.eval()
        collide_frank += collide_f
        ret_frank_list.append(dis)
        del frank_env

        # vx_np = np.array(vx).reshape((-1, 1))
        # vy_np = np.array(vy).reshape((-1, 1))
        # vxvy = np.concatenate((vx_np, vy_np), 1)
        # np.savetxt(dir + '/virtual/' + str(t) + '.txt', vxvy)
        # xnone_np = np.array(x_).reshape((-1, 1))
        # ynone_np = np.array(y_).reshape((-1, 1))
        # xy_none = np.concatenate((xnone_np,ynone_np), 1)
        # np.savetxt(dir + '/none/' + str(t) + '.txt', xy_none)
        # y_np = np.array(y).reshape((-1,1))
        # x_np = np.array(x).reshape((-1,1))
        # xy = np.concatenate((x_np,y_np), 1)
        # np.savetxt(dir + '/Ours/' + str(t) + '.txt', xy)
        # franky_np = np.array(frank_y).reshape((-1,1))
        # frankx_np = np.array(frank_x).reshape((-1,1))
        # xy_frank = np.concatenate((frankx_np, franky_np), 1)
        # np.savetxt(dir + '/frank/' + str(t) + '.txt', xy_frank)






        if draw:
            width = int(WIDTH + WIDTH_ALL)
            height = int(HEIGHT_ALL)
            grid = plt.GridSpec(height, width, wspace=20, hspace=20)


            plt.figure(1, figsize=(12, 6))
            # plt_srl = plt.subplot(1, 2, 2)
            # plt_none = plt.subplot(1, 2, 1)
            plt_none = plt.subplot(grid[0:int(HEIGHT_ALL), 0:int(WIDTH_ALL)])
            plt_srl = plt.subplot(grid[3:int(HEIGHT+3), int(WIDTH_ALL):width])
            # plt_noSRL = plt.subplot(1, 3, 3)
            plt_none.set_title('virtual')
            plt_srl.set_title('physical')
            # plt_noSRL.set_title('noSRL')

            plt_srl.axis('scaled')
            # plt_srl.set_xlim([0.0, WIDTH])
            # plt_srl.set_ylim([0.0, HEIGHT])
            plt_srl.axis([0.0, WIDTH, 0.0, HEIGHT])
            # plt_noSRL.axis([0.0, WIDTH, 0.0, HEIGHT])
            # plt_srl.axis('equal')
            # plt_none.axis('equal')
            plt_srl.set_yticks([])
            plt_srl.set_xticks([])

            plt_none.axis('scaled')
            plt_none.axis([0.0, WIDTH_ALL, 0.0, HEIGHT_ALL])
            plt_none.set_yticks([])
            plt_none.set_xticks([])
            # plt_none.set_xlim([0.0, WIDTH_ALL])
            # plt_none.set_ylim([0.0, HEIGHT_ALL])
            # plt_srl.scatter(np.array(x[0]), np.array(y[0]), label='SRL', s=1, c='r')
            # plt_srl.scatter(np.array(x_[0]), np.array(y_[0]), label='NONE', s=1, c='g')
            # plt_srl.scatter(np.array(frank_x[0]), np.array(frank_y[0]), label="FRANK", s=1, c='purple')
            # plt_none.scatter(np.array(vx), np.array(vy), s=1, c='b')
            # plt_srl.scatter(np.array(x[0]), np.array(y[i]), s=1, c='r', alpha=1.0 * math.exp(5 * (i / len(x) - 1.0)))
            c = [t * 0.5 for t in range(len(x))]
            plt_srl.scatter(np.array(x), np.array(y), s=0.2, c= c,cmap='Reds',
                            alpha=0.2, label='SRL')
            c = [t * 0.5 for t in range(len(x_))]
            plt_srl.scatter(np.array(x_), np.array(y_), s=0.2, c=c,cmap='Blues',
                            alpha=0.2, label='NONE')

            c = [t * 0.5 for t in range(len(frank_x))]
            plt_srl.scatter(np.array(frank_x), np.array(frank_y), c=c,s=0.2 , cmap='Greens',
                            alpha=0.2, label='FRANK')

            c = [t * 0.5 for t in range(len(vx))]
            plt_none.scatter(np.array(vx), np.array(vy), s=0.2, c=c, cmap='Reds',
                            alpha=0.2)
            # for i in range(len(x)):
            # #     # print(float(i/len(x)))
            #     plt_srl.scatter(np.array(x[i]), np.array(y[i]), s=0.2, c='crimson', alpha=1.0*math.exp(4*(i/len(x)-1.0)))
            # for i in range(len(x_)):
            #     plt_srl.scatter(np.array(x_[i]), np.array(y_[i]), s=0.2, c='limegreen',alpha=1.0*math.exp(4*(i/len(x_)-1.0)))
            # for i in range(len(frank_x)):
            #     plt_srl.scatter(np.array(frank_x[i]), np.array(frank_y[i]), s=0.2, c='dodgerblue',alpha=1.0*math.exp(4*(i/len(frank_x)-1.0)))
            # for i in range(len(vx)):
            #     plt_none.scatter(np.array(vx[i]), np.array(vy[i]), s=0.2, c='violet',alpha=1.0*math.exp(4*(i/len(vx)-1.0)))
            plt_srl.legend(markerscale=10)
            # plt_noSRL.scatter(np.array(x_), np.array(y_), s=1, c='r')
            plt_srl.scatter(env.obj_x_p, env.obj_y_p, s=50, marker='*', c='k')
            plt_none.scatter(env.obj_x, env.obj_y, s=50, marker='*', c='k')

            # plt_noSRL.scatter(WIDTH / 2.0, HEIGHT / 2.0, s=10)
            name = {1:"same_edge", 2:"random", 3:"shuffle", 4:"rand"}
            if ep is None:
                if not os.path.exists('./plot_result/pdf'):
                    os.makedirs('./plot_result/pdf')
                # pp = PdfPages('./plot_result/pdf/' + name[evalType]+ '_' + str(t) +'.pdf')
                plt.savefig('./plot_result/general/' + name[evalType]+ '_' + str(t) +'.png')
                # pp.savefig()
                # plt.clf()
                # plt_none.cla()
                # plt_srl.cla()
                plt.close()
                # pp.close()
            elif ep % 100 == 0:
                if not os.path.exists('./plot_result/%s/ep_%d' % (env_name, ep)):
                    os.makedirs('./plot_result/%s/ep_%d' % (env_name, ep))
                plt.savefig('./plot_result/%s/ep_%d/srl_%d.png' % (env_name, ep, t))
            plt.clf()
            plt_none.cla()
            plt_srl.cla()
                # plt_noSRL.cla()
            # print("retL", ret, "r_ :", r_, "dis: ", dis, "dis_ :" ,dis_)
        # if draw:
        #     plt.figure(1, figsize=(15, 5))
        #     plt_srl = plt.subplot(1, 3, 2)
        #     plt_none = plt.subplot(1, 3, 1)
        #     plt_noSRL = plt.subplot(1, 3, 3)
        #     plt_none.set_title('None')
        #     plt_srl.set_title('SRL')
        #     plt_noSRL.set_title('noSRL')
        #     plt_srl.axis([0.0, WIDTH, 0.0, HEIGHT])
        #     plt_noSRL.axis([0.0, WIDTH, 0.0, HEIGHT])
        #     # plt_srl.axis('equal')
        #     # plt_none.axis('equal')
        #     plt_none.axis([0.0, WIDTH_ALL, 0.0, HEIGHT_ALL])
        #     plt_srl.scatter(np.array(x), np.array(y), s=1, c='g')
        #     plt_none.scatter(np.array(vx), np.array(vy), s=1, c='b')
        #     plt_noSRL.scatter(np.array(x_), np.array(y_), s=1, c='r')
        #     plt_srl.scatter(WIDTH / 2.0, HEIGHT / 2.0, s=10)
        #     plt_noSRL.scatter(WIDTH / 2.0, HEIGHT / 2.0, s=10)
        #     if ep is None:
        #         if not os.path.exists('./plot_result/general'):
        #             os.makedirs('./plot_result/general')
        #         plt.savefig('./plot_result/general/' + str(t) + '_' + str(evalType) + '.png')
        #         plt.clf()
        #         plt_none.cla()
        #         plt_srl.cla()
        #     elif ep % 100 == 0:
        #         if not os.path.exists('./plot_result/%s/ep_%d' % (env_name, ep)):
        #             os.makedirs('./plot_result/%s/ep_%d' % (env_name, ep))
        #         plt.savefig('./plot_result/%s/ep_%d/srl_%d.png' % (env_name, ep, t))
        #         plt.clf()
        #         plt_none.cla()
        #         plt_srl.cla()
        #         plt_noSRL.cla()
        #     # print("retL", ret, "r_ :", r_, "dis: ", dis, "dis_ :" ,dis_)
    time_cost = time.time() - start_t

    gt_np = np.array(gt_list)
    gr_np = np.array(gr_list)
    gc_np = np.array(gc_list)
    gain_np = np.concatenate((gt_np, gr_np, gc_np)).reshape((3,-1)).transpose(1,0)


    print("collideNum:", collide_frank)
    flag = 1
    r_1 = sorted(ret_srl_list)
    r_2 = sorted(ret_none_list)
    r_3 = sorted(ret_frank_list)
    # length = len(r_1)
    # h = int(length/20)
    # tail=int(length*19/20+1)
    # r1 = r_1[h:tail]
    # r2 = r_2[h:tail]
    # r3 = r_3[h:tail]
    savetxt = np.concatenate((r_1,r_2,r_3)).reshape((3,-1))
    savetxt = savetxt.transpose(1,0)

    dir_ = "result_same_edge_pthird.txt"
    np.savetxt('result30_pthird_hist', gain_np)
    np.savetxt(dir_, savetxt)
    print("time:{:.2f}".format(time_cost), "action:", len(gt_list))
    return reward, r_none, distance_physical, dis_nosrl, angle_srl, angle_none, \
           flag, r_1, r_2, r_3, std_list1, std_list2, std_list3, gt_list, gr_list, gc_list, collide, collide_, collide_frank

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
    num = 5000
    # draw = True
    draw = False
    print('running ', num, " paths:")
    # reward, r_none, distance_physical, dis_nosrl, angle_srl, angle_none, flag, r_1, r_2,std_list1, std_list2, std_list3, gt_list, gr_list, gc_list, collide, collide_ = PassiveHapticRdwEvaluate(actor_critic, args.seed,
    #                                      1, 0.99, None, None, 10, None, 0, args.env_name, False, num, draw, evalType=3)
    # print("TYPE1:")
    # print("SRL:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\t".format(distance_physical/num, r_1[0], r_1[int(num/2)], r_1[num-1], angle_srl.item()/num))
    # print("None:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\t".format(dis_nosrl/num, r_2[0], r_2[int(num/2)], r_2[num-1], angle_none.item()/num))

    # reward, r_none, distance_physical, dis_nosrl, angle_srl, angle_none, flag, r_1, r_2,std_list1, std_list2, std_list3, gt_list, gr_list, gc_list, collide, collide_ = PassiveHapticRdwEvaluate(actor_critic, args.seed,
    #                                      1, 0.99, None, None, 10, None, 0, args.env_name, False, num, draw, evalType=1)
    # print("TYPE2:")
    # print("SRL:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\t".format(distance_physical/num, r_1[0], r_1[int(num/2)], r_1[num-1], angle_srl.item()/num))
    # print("None:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\t".format(dis_nosrl/num, r_2[0], r_2[int(num/2)], r_2[num-1], angle_none.item()/num))


    reward, r_none, distance_physical, dis_nosrl, angle_srl, angle_none, flag, r_1, r_2, r_3, std_list1, std_list2, std_list3, gt_list, gr_list, gc_list, collide, collide_, collide_frank = PassiveHapticRdwEvaluateFrank(actor_critic, args.seed,
                                         1, 0.99, None, None, 10, None, 0, args.env_name, False, num, draw, evalType=2)
    len = len(r_1)
    left = int(len/4)
    right = int(len*3/4)
    print("TYPE3:")
    print("SRL:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\tIQR:{:.2f}".format(distance_physical/num, r_1[0], r_1[int(num/2)], r_1[num-1], angle_srl.item()/num,r_1[right]-r_1[left] ), "\tcollide", collide)
    print("None:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\tanglr:{:.2f}\tIQR:{:.2f}".format(dis_nosrl/num, r_2[0], r_2[int(num/2)], r_2[num-1], angle_none.item()/num, r_2[right]-r_2[left]), "\tcollide", collide_)
    print("FRANK:\tdistance:{:.2f}\t{:2f}|{:.2f}|{:.2f}\t\t\tIQR:{:.2f}".format(0.0, r_3[0], r_3[int(num/2)], r_3[num-1], r_3[right]-r_3[left]), "\tcollide", collide_frank)