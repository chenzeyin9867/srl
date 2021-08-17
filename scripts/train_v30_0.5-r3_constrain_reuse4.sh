
python main.py \
    --log-interval 50 --num-steps 2048 --num-processes 1 \
    --entropy-coef 0  --ppo-epoch 4  --clip-param 0.1\
    --num-mini-batch 4 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 20000000 \
    --use-linear-lr-decay --env-name 624_train_v30_0.5-r3_constrain_reuse4_initstd0.5