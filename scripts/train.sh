python main.py \
    --log-interval 20 --num-steps 8192 --num-processes 1 --lr 3e-4 \
    --entropy-coef 0.001  --ppo-epoch 8  --clip-param 0.1\
    --num-mini-batch 4 --gamma 0.99 --gae-lambda 0.80 --num-env-steps 80000000 \
    --use-linear-lr-decay --env-name 927_r_-r2-r3_old_div10_entropy0.001_reuse8_penalty10_clip0.1_tanh_final10*1-disexp1_obs13_gainNoThreshold_gamma0.99_buf8192_bsz2048_hs256_lr3e-4
    # --use-linear-lr-decay --env-name 913_r_1-r2+r3_exp0.5_div10_entropy0.01_reuse8_penalty5_clip0.1_tanh_final10*1-disexp2_obs13_gainNoThreshold_gamma0.99_buf2048     



## hidden state only can be used as 128
## learning rate can't exceed 3e-4
