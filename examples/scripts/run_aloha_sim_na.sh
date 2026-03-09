#!/bin/bash
# ============================================================
# DSRL-NA Training Script for Aloha Sim
# ============================================================

# --- Device Configuration ---
# Single GPU (default)
# device_id=0
# export local_policy_device=0
# export dsrl_device_idx=0
# export kv_cache_device=0
# export CUDA_VISIBLE_DEVICES=$device_id

# Multi-GPU (uncomment and adjust)
device_id=0,1,2,3
export local_policy_device=0,1,2,3
export dsrl_device_idx=0
export kv_cache_device=0
export CUDA_VISIBLE_DEVICES=$device_id

# --- Environment ---
export DISPLAY=:0
export MUJOCO_GL=egl

# --- Project / Logging ---
proj_name=DSRL
export EXP=./logs/$proj_name
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --- Policy Serving Mode ---
# Use local policy (1) or websocket remote policy (0)
use_local_policy=1
save_kv_cache=1
put_kv_cache_on_cpu=1  # Saves GPU memory; overwrites kv_cache_device

# Remote policy settings (only used when use_local_policy=0)
export remote_host="localhost"
export remote_port="8090"

# --- Environment / Action Configuration ---
query_freq=50
max_timesteps=400
action_chunk_size=50
action_magnitude=0.75

# --- Training Hyperparameters ---
batch_size=128
max_steps=1000000
multi_grad_step=20
num_initial_traj_collect=125
online_buffer_size=150000
num_qs=10

# ---- Logging and Evaluation ---
log_interval=100
eval_interval=5000
eval_episodes=10
checkpoint_interval=10000

# --- NA-Specific Hyperparameters ---
flow_integration_steps=10
action_critic_steps=15
noise_critic_steps=5
noise_actor_steps=5
train_all_together=0  # If 1, UTD = multi_grad_step. if 0, UTD = action_critic_steps + noise_critic_steps

# --- Exploration ---
grl_noise_sample=1
backup_entropy=0
noise_scale_inside=0

# --- Optional ---
restore_path=""

# ============================================================

python3 examples/launch_train_aloha_sim_na.py \
--env aloha_cube \
--prefix dsrl_pi0_aloha_sim_na_${num_initial_traj_collect}traj_${num_qs}qs_${batch_size}bs_${grl_noise_sample}grl \
--wandb_project ${proj_name} \
--batch_size ${batch_size} \
--discount 0.999 \
--seed 0 \
--max_steps ${max_steps} \
--eval_interval ${eval_interval} \
--eval_episodes ${eval_episodes} \
--log_interval ${log_interval} \
--checkpoint_interval ${checkpoint_interval} \
--multi_grad_step ${multi_grad_step} \
--resize_image 64 \
--action_magnitude ${action_magnitude} \
--query_freq ${query_freq} \
--num_qs ${num_qs} \
--hidden_dims 128 \
--target_entropy 0.0 \
--num_initial_traj_collect ${num_initial_traj_collect} \
--action_chunk_size ${action_chunk_size} \
--max_timesteps ${max_timesteps} \
--restore_path "${restore_path}" \
--use_local_policy ${use_local_policy} \
--save_kv_cache ${save_kv_cache} \
--backup_entropy ${backup_entropy} \
--grl_noise_sample ${grl_noise_sample} \
--flow_integration_steps ${flow_integration_steps} \
--online_buffer_size ${online_buffer_size} \
--action_critic_steps ${action_critic_steps} \
--noise_critic_steps ${noise_critic_steps} \
--noise_actor_steps ${noise_actor_steps} \
--train_all_together ${train_all_together} \
--put_kv_cache_on_cpu ${put_kv_cache_on_cpu} \
--noise_scale_inside ${noise_scale_inside}
