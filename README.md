<div align="center">

# DSRL for π₀: Diffusion Steering via Reinforcement Learning

## [[website](https://diffusion-steering.github.io)]      [[paper](https://arxiv.org/abs/2506.15799)]

</div>


## Overview
This repository provides the official implementation for our paper: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799) (CoRL 2025).

Specifically, it contains a JAX-based implementation of DSRL (Diffusion Steering via Reinforcement Learning) for steering a pre-trained generalist policy, [π₀](https://github.com/Physical-Intelligence/openpi), across various environments, including:

- **Simulation:** Libero, Aloha  
- **Real Robot:** Franka

If you find this repository useful for your research, please cite:

```
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

## Installation
1. Create a conda environment:
```
conda create -n dsrl_pi0 python=3.11.11
conda activate dsrl_pi0
```

2. Clone this repo with all submodules
```
git clone git@github.com:nakamotoo/dsrl_pi0.git --recurse-submodules
cd dsrl_pi0
```

3. Install all packages and dependencies
```
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12]==0.5.0"

# install openpi
pip install -e openpi
pip install -e openpi/packages/openpi-client

# install Libero
pip install -e LIBERO
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu # needed for libero
```

## Training (Simulation)
Libero
```
bash examples/scripts/run_libero.sh
```
Aloha
```
bash examples/scripts/run_aloha.sh
```
### Training Logs
We provide sample W&B runs and logs: https://wandb.ai/mitsuhiko/DSRL_pi0_public

## Training (NA — Noise aliasing variant)
The Noise-Aliasing (NA) variant steers the pre-trained policy by learning an action critic in the **environment action space** and then distill that into a noise space critic.

Aloha simulation:
```
bash examples/scripts/run_aloha_sim_na.sh
```

### Key NA Configuration Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_local_policy` | `1` | Use local policy (`1`) or remote websocket (`0`) |
| `put_kv_cache_on_cpu` | `1` | Offload KV cache to CPU to save GPU memory |
| `flow_integration_steps` | `10` | Euler integration steps for flow policy |
| `action_critic_steps` | `20` | Action critic gradient steps per training cycle |
| `noise_critic_steps` | `10` | Noise critic gradient steps per training cycle |
| `noise_actor_steps` | `5` | Noise actor gradient steps per training cycle |
| `grl_noise_sample` | `0` | Sample half of distillation noise from the noise actor |
| `train_all_together` | `0` | If `1`, train all components jointly (UTD = `multi_grad_step`) |
| `noise_scale_inside` | `0` | If `1`, noise scaling inside tanh distribution; if `0`, external scaling |
| `query_freq` | `50` | How often (in env steps) to query the noise actor |
| `action_chunk_size` | `50` | Number of actions per chunk sent to the environment |
| `max_timesteps` | `400` | Maximum environment steps per episode |

## Training (Real)
For real-world experiments, we use the remote hosting feature from pi0 (see [here](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)) which enables us to host the pi0 model on a higher-spec remote server, in case the robot's client machine is not powerful enough. 

0. Setup Franka robot and install DROID package [[link](https://github.com/droid-dataset/droid.git)]

1. [On the remote server] Host pi0 droid model on your remote server
```
cd openpi && python scripts/serve_policy.py --env=DROID
```
2. [On your robot client machine] Run DSRL
```
bash examples/scripts/run_real.sh
```


## Credits
This repository is built upon [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR) repositories. 
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at nakamoto\[at\]berkeley\[dot\]edu 
