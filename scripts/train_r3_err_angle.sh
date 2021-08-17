python main.py \
    --log-interval 10 --num-steps 2048 --num-processes 1 \
    --entropy-coef 0  --ppo-epoch 8  --clip-param 0.1\
    --num-mini-batch 4 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 20000000 \
    --use-linear-lr-decay --env-name 623_train_v30_r3_err_angle_f_1-0.05d-0.5a 