import os
import time
from collections import deque
from tensorboardX import SummaryWriter
import numpy as np
import torch
import math
from a2c_ppo_acktr import algo, myutils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs_general import  make_passive_haptics_env, RunningStats, OBS_NORM, REWARD_SCALE
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import PassiveHapticRdwEvaluate
from a2c_ppo_acktr.distributions import FixedNormal

# OBS_NORM = False
def main():
    args = get_args()
    print(args)
    writer1 = SummaryWriter('runs/' + args.env_name)
    if not os.path.exists('runs/' + args.env_name):
        os.makedirs('runs/' + args.env_name)
    torch.manual_seed(args.seed)
    flag = 0
    R_none = 0
    Dis_none = 0
    # RunningState = RunningStats()
    # RewardState =  RunningStats()
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    #envs = make_rdw_env(args.seed, args.num_processes, args.gamma, args.log_dir, device, 10)
    envs = make_passive_haptics_env(args.seed, args.num_processes, args.gamma, args.log_dir, device, 10)
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space)
    if args.load_epoch != 0:
        actor_critic = \
            torch.load('./trained_models/' + args.env_name + '/%d.pth' % args.load_epoch)
        print("Loading the " + args.env_name + '/_%d.pt' % args.load_epoch + ' to train')
    actor_critic.to(device)
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    t_start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes


    min_median = 10000
    min_median_epoch = 0
    for j in range(args.load_epoch, num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            myutils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action_mean, action_std = actor_critic.act(rollouts.obs[step])
                # print(action_std)
                dist = FixedNormal(action_mean, action_std)
                action = dist.sample()
                # print(action_mean, action_std, action)
                # action = action_mean
                # action_log_prob = dist.log_probs(action)
                # action = action.clamp(-1.0, 1.0)

                action_log_prob = dist.log_probs(action)
                # print(action)
                # action = action.clamp(-1.0, 1.0)
            # print(step)
            # Obser reward and next obs
            obs, reward, done, infos, ret_reward = envs.step(action)
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in infos])
            rollouts.insert(obs, None, action,
                            action_log_prob, value, torch.Tensor([reward]), masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, entropy_loss, total_loss = agent.update(rollouts, args)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.env_name)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save(actor_critic, os.path.join(save_path,  "%d.pth" % j))

        num = 100
        mid = int(num/2)
        l = int(num/4)
        r = int(num*3/4)
        if j % args.log_interval == 0:
            r_eval, r_none, distance, disnosrl, angle_srl, angle_none, flag, m1, m2, std1, std2, std3, gt, gr, gc, c, c_ = \
                PassiveHapticRdwEvaluate(actor_critic, args.seed,
                                         args.num_processes, args.gamma, args.log_dir, device, 10, j, flag, args.env_name, num=num)
            # print("Epoch_%d" % j, "\t|eval_rewards:{:.2f}".format(r_eval))
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            if R_none == 0:
                R_none = r_none
                Dis_none = disnosrl
            if min_median > m1[mid]:
                min_median = m1[mid]
                min_median_epoch = j
            print(
                # "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}\n"
                "Epoch_%d/%d" % (j, num_updates), "\t|v_loss:{:.4f}  |a_loss:{:.4f}|entropy:{:.4f} |total_l:{:.4f}"
                                                  " |r_eval:{:.2f} |r_none:{:.2f} |srl_dis:{:.2f} |no_dis:{:.2f} "
                                                  "|ang_srl:{:.2f} |ang_none:{:.2f} |srl:{:.2f}"
                                                  " |none:{:.2f}"
                .format(value_loss, action_loss, entropy_loss, total_loss, r_eval.item(), R_none.item(),
                        distance, Dis_none, angle_srl.item(), angle_none.item(), m1[mid], m2[mid]))
            print("std:{:.4f} {:.4f} {:.4f}\t|gt:{:.2f}\t|gr:{:.2f}\t|gc:{:.2f}\t|".
                  format(np.mean(std1), np.mean(std2), np.mean(std3), np.mean(gt).item(), np.mean(gr).item(),
                         np.mean(gc).item()),
                  "srl_reset:", c, " reset:", c_, " min_epoch:", min_median_epoch,
                  "min_median:{:.2f} |t:{:.2f} |quart_srl:{:.2f} no_quart:{:.2f}"
                    .format(min_median, end - t_start, m1[r]-m1[l], m2[r]-m2[l]))
            t_start = time.time()
            # print(
            #     # "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}\n"
            #     "Epoch_%d/%d" % (j, num_updates), "\t|mean_r:{:.2f}\t|v_loss:{:.8f}\t|a_loss:{:.4f}\t|t:{:.2f}"
            #     .format(np.mean(episode_rewards), value_loss, action_loss, end - t_start))
        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     #ob_rms = visualize.get_vec_normalize(envs).ob_rms
        #     # evaluate(actor_critic, ob_rms, args.env_name, args.seed,
        #     #          args.num_processes, eval_log_dir, device)
        #
        #     # eval_env = make_rdw_env(args.seed, args.num_processes, args.gamma, args.log_dir, device, 10)
        #     r_eval = rdw_evaluate(actor_critic, args.seed, args.num_processes, args.gamma, args.log_dir, device, 10)
        #     print("Epoch:", j, "\teval_rewards:{:.2f}".format(r_eval))
        writer1.add_scalar('v_loss', value_loss, global_step=j)
        writer1.add_scalar('ac_loss', action_loss, global_step=j)
        writer1.add_scalar('entropy_loss', entropy_loss, global_step=j)
        writer1.add_scalar('total_loss', total_loss, global_step=j)
        writer1.add_scalar('dis', distance, global_step=j)
        writer1.add_scalar('eval_reward', r_eval, global_step=j)
        writer1.add_scalar('err_angle', angle_srl, global_step=j)
        writer1.add_scalar('median_err_dis', m1[mid], global_step=j)
if __name__ == "__main__":
    main()

