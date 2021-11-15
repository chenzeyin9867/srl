# Introduction
Implementation of the paper A Steering Algorithm for Redirected Walking Using Reinforcement Learning

# Acknowledgment

This repo is a basic implementation of the paper [A Steering Algorithm for Redirected Walking Using Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/8998570). Limited by ability and time, I cannot guarantee that the code is completely consistent with the origin paper, only the basic framework can be provided.
This implementation is based on fantastic implementaion [PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) by Kostrikov.

# Training
* Update your env configuaration like [w, h] of your physical space in srl_core/envs_general.py
* Fine-tune your hyper-parameters in ```args/train.txt```
* Training a new model  
    ```bash
    python main.py --config args/train.txt
    ```
# Evaluation and plot the heatmap
* ```bash
  python test_heatmap.py --config args/eval.txt --load_epoch $Your_MODEL
  ```  