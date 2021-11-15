import os
import time
from tensorboardX import SummaryWriter
import numpy as np
import torch
import tqdm
from core import algo, myutils
from core.arguments import get_args
from core.envs_general import  PassiveHapticsEnv
from core.model import Policy
from core.storage import RolloutStorage
from evaluation import srlEvaluate
from core.distributions import FixedNormal

# OBS_NORM = False
def main():
    args = get_args()
    params = args.__dict__
    writer1 = SummaryWriter('runs/' + args.env_name)
    if not os.path.exists('runs/' + args.env_name):
        os.makedirs('runs/' + args.env_name)
    
    
    # Experimental settings
    
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)    
    device = torch.device("cuda:0" if args.cuda else "cpu")

    flag = 0
    # debug infos
    print(device)
    print(args)

    # Instance the env, training batch
    envs = PassiveHapticsEnv(args.gamma, args.stack_frame, eval=False)
    
    # The actor module
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space)

    if args.load_epoch != 0:
        actor_critic = torch.load('./trained_models/' + args.env_name + '/%d.pth' % args.load_epoch)
        print("Loading the " + args.env_name + '/_%d.pt' % args.load_epoch + ' to train')
    actor_critic.to(device)
    
    # The PPO module
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
    
    # Buffer for updating
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    
    # Initialize the Env
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    t_start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    r_eval = 0
    gt, gr, gc, c = 0, 0, 0, 0

    for j in tqdm.trange(args.load_epoch, num_updates):

        if args.use_linear_lr_decay:
            # Update the lr
            myutils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action_mean, action_std = actor_critic.act(rollouts.obs[step])
                # print(action_std)
                dist = FixedNormal(action_mean, action_std)
                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)
                action_log_prob = dist.log_probs(action)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in infos])
            rollouts.insert(obs, action, action_log_prob, value, torch.Tensor([reward]), masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, entropy_loss, total_loss, explained_variance = agent.update(rollouts, args)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.env_name)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save(actor_critic, os.path.join(save_path,  "%d.pth" % j))

        num = 100

        if j % args.log_interval == 0:
            r_eval, r_none, flag, std1, std2, std3, gt, gr, gc, c, c_, l, l_ = srlEvaluate(actor_critic, j, flag, **params)
            
            print(args.env_name)
            print(
                "Epoch_%d/%d" % (j, num_updates),"\tv_loss:{:.4f}\ta_loss:{:.4f}\te_loss:{:.4f}\tr_phrl:{:.2f}\tr_none:{:.2f}"
                    .format(value_loss, action_loss, entropy_loss, r_eval.item(), r_none.item()))
            print("std:{:.3f} {:.3f} {:.3f}\t" 
                  "|gt:{:.2f}|gr:{:.2f} |gc:{:.2f}\t|".
                  format(np.mean(std1), np.mean(std2), np.mean(std3), 
                  np.mean(gt).item(), np.mean(gr).item(), np.mean(gc).item()),
                  "reset_phrl:", c, " reset_none:", c_,  "\tSRL_dis:{:.4f}\tNone_dis:{:.4f}\t|t:{:.2f} "
                    .format(l, l_, time.time() - t_start)) 
            t_start = time.time()
        writer1.add_scalar('value_loss', value_loss, global_step=j)
        writer1.add_scalar('actor_loss', action_loss, global_step=j)
        writer1.add_scalar('entropy_loss', entropy_loss, global_step=j)
        writer1.add_scalar('total_loss', total_loss, global_step=j)
        # writer1.add_scalar('physical_distance_error', distance, global_step=j)
        writer1.add_scalar('phrl_reward', r_eval, global_step=j)
        # writer1.add_scalar('physical_angle_error', angle_srl, global_step=j)
        # writer1.add_scalar('median_distance_error', m1[mid], global_step=j)
        writer1.add_scalar('gt', np.mean(gt).item(), global_step=j)
        writer1.add_scalar('gr', np.mean(gr).item(), global_step=j)
        writer1.add_scalar('gc', np.mean(gc).item(), global_step=j)
        writer1.add_scalar('reset', c, global_step=j)
        writer1.add_scalar("explained_var", explained_variance, global_step=j)
        writer1.add_scalar("SRL_distance", l, global_step=j)
        
if __name__ == "__main__":
    main()

