"""Training utilities for DSRL-NA (Aloha Sim)."""
from copy import deepcopy
from examples.train_utils_sim import obs_to_img, obs_to_pi_zero_input, obs_to_qpos
from jaxrl2.agents.pixel_sac import PixelSACLearnerNA
from collections import deque
import os
import time
import collections
from tqdm import tqdm
import numpy as np
import jax
from openpi_client import image_tools
from videoio import videosave
from flax.core import frozen_dict
from termcolor import colored, cprint
import wandb

from examples.train_utils_na import add_online_data_to_buffer_na, generate_distillation_batch, get_next_actions_from_dp, remove_original_obs_keys

def trajwise_alternating_training_loop_na(variant, agent: PixelSACLearnerNA, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       shard_fn=None, agent_dp=None, robot_config=None):
    all_devices = jax.devices()
    # local_policy_sharding = None
    agent_dp_device_proxy = None
    dsrl_device = None
    if os.environ.get("local_policy_device") is not None:
        local_policy_device = os.environ['local_policy_device'].split(',')
        local_policy_device = [int(x) for x in local_policy_device]
        selected_devices = [all_devices[i] for i in local_policy_device]
        agent_dp_device_proxy = selected_devices[0]
        # local_policy_mesh = jax.sharding.Mesh(selected_devices, ("x",))
        # local_policy_sharding = jax.sharding.NamedSharding(local_policy_mesh, jax.sharding.PartitionSpec())
    if os.environ.get("dsrl_device_idx") is not None:
        dsrl_device_idx = int(os.environ['dsrl_device_idx'])
        dsrl_device = all_devices[dsrl_device_idx]
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    # if shard_fn is not None:
    #     replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)
    keys_to_remove = ['original_observations', 'original_next_observations']
    cprint(f"Distillation noise sampling strategy: {variant['grl_noise_sample']=}", "green" if variant['grl_noise_sample'] else "red")
    i = 0
    total_env_steps = 0
    total_num_traj = 0
    moving_avg_ep_len = collections.deque(maxlen=20)
    moving_avg_success = collections.deque(maxlen=20)
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
    action_critic_updates = 0
    noise_critic_updates = 0
    noise_actor_updates = 0
    scalar_metric_sums = collections.defaultdict(float)
    scalar_metric_counts = collections.defaultdict(int)
    with tqdm(total=variant.max_steps, initial=0, dynamic_ncols=True) as pbar:
        while i <= variant.max_steps:
            sample_from_agent = i != 0 or variant['restore_path'] != ''
            traj = collect_traj_na(variant, agent, env, i, sample_from_agent, agent_dp, wandb_logger, total_num_traj, robot_config,
                                   moving_avg_success=moving_avg_success, moving_avg_ep_len=moving_avg_ep_len)
            total_num_traj += 1
            add_online_data_to_buffer_na(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            tqdm.write(f'online buffer timesteps length: {len(online_replay_buffer)}')
            tqdm.write(f'online buffer num traj: {total_num_traj}')
            tqdm.write(f'total env steps: {total_env_steps}')
            
            # total_inter_steps = (variant.action_critic_steps + variant.noise_critic_steps + variant.noise_actor_steps)
            if variant['train_all_together']:
                total_inter_steps = variant.multi_grad_step
            else:
                total_inter_steps = (variant.action_critic_steps + variant.noise_critic_steps)

            if i == 0:
                num_gradsteps = min(3000,len(online_replay_buffer) * total_inter_steps)
            else:
                num_gradsteps = len(traj["rewards"]) * total_inter_steps
            tqdm.write(f'num_gradsteps: {num_gradsteps}')
            times = collections.defaultdict(list)
            if total_num_traj >= variant.num_initial_traj_collect:
                for grad_idx in tqdm(range(num_gradsteps), dynamic_ncols=True, leave=False):
                    if i == 0:
                        perform_control_eval_na(agent, eval_env, i, variant, wandb_logger, agent_dp, robot_config)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    t0 = time.time()
                    if variant['train_all_together']:
                        train_action_critic = True
                        train_noise_critic = True
                        train_noise_actor = True
                    else:
                        inter_step = grad_idx % total_inter_steps
                        train_action_critic = (inter_step < variant.action_critic_steps)
                        train_noise_critic = (inter_step >= variant.action_critic_steps) and (inter_step < variant.action_critic_steps + variant.noise_critic_steps)
                        # Train noise actor separately after action critic and noise critic updates
                        # train_noise_actor = (inter_step >= variant.action_critic_steps + variant.noise_critic_steps)
                        # Train action critic and noise actor in the same step but only after the first round of noise and action critic updates
                        train_noise_actor = (inter_step < variant.noise_actor_steps) and (grad_idx // total_inter_steps) > 0
                    
                    start = time.time()
                    batch = next(replay_buffer_iterator)
                    times['data_time'].append(time.time() - start)
                    
                    start = time.time()
                    # Split batch into jittable and non jittable based on keys_to_remove
                    jittable_batch = {k: v for k, v in batch.items() if k not in keys_to_remove}
                    non_jittable_batch = {k: v for k, v in batch.items() if k in keys_to_remove}
                    times['preprocess_time'].append(time.time() - start)
                    
                    start = time.time()
                    if shard_fn is not None:
                        jittable_batch = shard_fn(jittable_batch)
                    times['shard_time'].append(time.time() - start)
                    
                    batch = {**jittable_batch, **non_jittable_batch} 
                    
                    if train_noise_critic:
                        # Fetch cache for current obs
                        # if 'indices' in batch:
                        indices = batch['indices']
                        original_k_cache = replay_buffer.get_cache(indices, 'k')
                        original_v_cache = replay_buffer.get_cache(indices, 'v')
                        if agent_dp_device_proxy is not None:
                            original_k_cache = jax.device_put(original_k_cache, agent_dp_device_proxy)
                            original_v_cache = jax.device_put(original_v_cache, agent_dp_device_proxy)
                        batch['original_k_cache'] = original_k_cache
                        batch['original_v_cache'] = original_v_cache
                        start = time.time()
                        distill_noise, distill_actions, end_times = generate_distillation_batch(batch, agent, agent_dp, robot_config, variant, agent_dp_device=agent_dp_device_proxy)
                        if dsrl_device is not None:
                            distill_noise = jax.device_put(distill_noise, dsrl_device)
                            distill_actions = jax.device_put(distill_actions, dsrl_device)
                        distill_batch = frozen_dict.freeze({
                            'distill_noise': distill_noise,
                            'distill_actions': distill_actions,
                        })
                        times['distill_batch_time'].append(time.time() - start)
                        [times[f'distill_batch_{k}'].append(v) for k, v in end_times.items()]
                    else:
                        distill_batch = None
                    
                    start = time.time()
                    # Remove original_obs keys before update (not JIT-compatible due to string prompts)
                    batch_for_update = remove_original_obs_keys(batch)
                    times['remove_keys_time'].append(time.time() - start)
                    if train_action_critic:
                        start = time.time()
                        next_actions_noise, next_actions_noise_log_probs = agent.sample_actions_with_log_probs(batch['next_observations'])
                        # next_actions_noise = agent.sample_actions(batch['next_observations'])
                        times['sample_next_actions_time'].append(time.time() - start)

                        start = time.time()
                        # Fetch cache for next obs
                        # if 'next_indices' in batch:
                        next_indices = batch['next_indices']
                        original_next_k_cache = replay_buffer.get_cache(next_indices, 'k')
                        original_next_v_cache = replay_buffer.get_cache(next_indices, 'v')
                        if agent_dp_device_proxy is not None:
                            original_next_k_cache = jax.device_put(original_next_k_cache, agent_dp_device_proxy)
                            original_next_v_cache = jax.device_put(original_next_v_cache, agent_dp_device_proxy)
                        batch['original_next_k_cache'] = original_next_k_cache
                        batch['original_next_v_cache'] = original_next_v_cache

                        agent_dp_output_batched, end_times = get_next_actions_from_dp(agent_dp, batch, next_actions_noise, agent.action_chunk_shape, robot_config, variant)
                        times['next_actions_time'].append(time.time() - start)
                        [times[f'next_actions_{k}'].append(v) for k, v in end_times.items()]

                        batch_for_update = dict(batch_for_update)
                        if dsrl_device is not None:
                            agent_dp_output_batched = jax.device_put(agent_dp_output_batched, dsrl_device)
                        batch_for_update["next_executed_actions"] = agent_dp_output_batched
                        batch_for_update["next_log_probs"] = next_actions_noise_log_probs
                        batch_for_update = frozen_dict.freeze(batch_for_update)
                    
                    start = time.time()
                    update_info = agent.update(batch_for_update, distill_batch=distill_batch, train_action_critic=train_action_critic, train_noise_actor=train_noise_actor)
                    update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                    for k, v in update_info.items():
                        if np.asarray(v).ndim == 0:
                            scalar_v = float(v)
                            if np.isfinite(scalar_v):
                                scalar_metric_sums[k] += scalar_v
                                scalar_metric_counts[k] += 1
                    times['update_time'].append(time.time() - start)
                    times['total_time'].append(time.time() - t0)

                    pbar.update()
                    i += 1
                    action_critic_updates += int(train_action_critic)
                    noise_critic_updates += int(train_noise_critic)
                    noise_actor_updates += int(train_noise_actor)
                    
                    if i % variant.log_interval == 0 or (i == 1 or i == variant.max_steps):
                        if scalar_metric_counts:
                            averaged_metrics = {
                                f'training/{k}': scalar_metric_sums[k] / scalar_metric_counts[k]
                                for k in scalar_metric_counts.keys()
                            }
                            wandb_logger.log(averaged_metrics, step=i)
                            scalar_metric_sums = collections.defaultdict(float)
                            scalar_metric_counts = collections.defaultdict(int)

                        for k, v in update_info.items():
                            if np.asarray(v).ndim <= 2 and np.asarray(v).ndim > 0:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)
                        wandb_logger.log({
                            'num_updates/action_critic_updates': action_critic_updates,
                            'num_updates/noise_critic_updates': noise_critic_updates,
                            'num_updates/noise_actor_updates': noise_actor_updates,
                            'num_updates/total_updates': action_critic_updates + noise_critic_updates + noise_actor_updates,
                        }, step=i)
                        if times:
                            avg_times = {f'train_time/{k}': np.mean(v) for k, v in times.items()}
                            wandb_logger.log(avg_times, step=i)
                            times = collections.defaultdict(list)

                    if i % variant.eval_interval == 0 or (i == 1 or i == variant.max_steps):
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': total_num_traj}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if i % variant.eval_interval == 0:
                            perform_control_eval_na(agent, eval_env, i, variant, wandb_logger, agent_dp, robot_config)
                            if hasattr(agent, 'perform_eval'):
                                agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and (i % variant.checkpoint_interval == 0 or (i == 1 or i == variant.max_steps)):
                        agent.save_checkpoint(variant.outputdir, i, None, keep=2)


def perform_control_eval_na(agent, eval_env, step, variant, wandb_logger, agent_dp, robot_config):
    """Run evaluation rollouts for the NA pipeline and log metrics + videos to wandb."""
    query_frequency = variant.query_freq
    max_timesteps = robot_config['max_timesteps']
    chunk_size = robot_config['action_chunk_size']
    use_local_policy = robot_config['use_local_policy']
    env_max_reward = variant.env_max_reward

    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed + 456)

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            last_obs = eval_env.reset()
        elif 'aloha' in variant.env:
            last_obs, _ = eval_env.reset()

        obs_pi_zero = obs_to_pi_zero_input(last_obs, variant)
        image_list = []
        rewards = []

        for t in tqdm(range(max_timesteps), desc=f'Eval {rollout_id+1}/{variant.eval_episodes}', dynamic_ncols=True, leave=False):
            curr_image = obs_to_img(last_obs, variant)
            image_list.append(curr_image)

            if t % query_frequency == 0:
                rng, key = jax.random.split(rng)

                if use_local_policy:
                    output = agent_dp.get_prefix_rep_and_kv_cache(obs_pi_zero)
                    kv_cache = output["kv_cache"]
                    img_rep_pi0 = output['prefix_rep'][0]
                else:
                    output = agent_dp.get_prefix_rep(obs_pi_zero)
                    img_rep_pi0 = output['prefix_rep'][0]
                    kv_cache = None

                qpos = obs_to_qpos(last_obs, variant)
                qpos = np.concatenate([qpos, img_rep_pi0])

                if variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    }

                if step == 0:
                    # For initial evaluation, sample from standard gaussian noise to evaluate base policy
                    noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                    noise_repeat = jax.numpy.repeat(noise[:, -1:, :], chunk_size - noise.shape[1], axis=1)
                    noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                else:
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = np.repeat(actions_noise[-1:, :], chunk_size - actions_noise.shape[0], axis=0)
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

                agent_dp_output = agent_dp.infer(obs_pi_zero, noise=np.asarray(noise), kv_cache=kv_cache)
                action = agent_dp_output["actions"]

            action_t = action[t % query_frequency]

            if 'libero' in variant.env:
                last_obs, reward, done, _ = eval_env.step(action_t)
            elif 'aloha' in variant.env:
                last_obs, reward, terminated, truncated, _ = eval_env.step(action_t)
                done = terminated or truncated

            obs_pi_zero = obs_to_pi_zero_input(last_obs, variant)
            rewards.append(reward)
            if done:
                break

        # Per episode stats
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)

        tqdm.write(f'Eval rollout {rollout_id}: {episode_return=}, Success: {is_success}, Len: {t+1}')
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50)}, step=step)

    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nEval @ step {step} — Success rate: {success_rate}\nAverage return: {avg_return}\nAvg episode len: {avg_episode_len}\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=step)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=step)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=step)
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=step)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    tqdm.write(summary_str)


def collect_traj_na(variant, agent, env, i, sample_from_agent, agent_dp=None, wandb_logger=None, traj_id=None, robot_config=None, moving_avg_success=None, moving_avg_ep_len=None):
    all_devices = jax.devices("gpu")
    kv_cache_device = None
    if os.environ.get("kv_cache_device") is not None:
        kv_cache_device_idx = int(os.environ["kv_cache_device"])
        kv_cache_device = all_devices[kv_cache_device_idx]

    put_kv_cache_on_cpu = variant['put_kv_cache_on_cpu']
    query_frequency = variant.query_freq
    max_timesteps = robot_config['max_timesteps']
    chunk_size = robot_config['action_chunk_size']
    use_local_policy = robot_config["use_local_policy"]
    save_kv_cache = robot_config["save_kv_cache"]
    if save_kv_cache:
        assert use_local_policy, "save_kv_cache is true, but use_local_policy is false"
    agent._rng, rng = jax.random.split(agent._rng)
    if 'libero' in variant.env:
        last_obs = env.reset()
    elif 'aloha' in variant.env:
        last_obs, _ = env.reset()
    obs_pi_zero = obs_to_pi_zero_input(last_obs, variant)
    tqdm.write(colored(f"Starting Trial {traj_id}. Using success threshold: {variant['success_threshold']}. Sample from agent: {sample_from_agent}", 'yellow' if not sample_from_agent else 'green'))
    
    rewards = []
    action_list = []
    executed_action_list = []
    obs_list = []
    original_obs_list = []
    image_list = []
    # Store kv_cache for each inference step
    k_cache_outs = []
    v_cache_outs = []

    try:
        for t in tqdm(range(max_timesteps), dynamic_ncols=True):
            curr_image = obs_to_img(last_obs, variant)
            image_list.append(curr_image)
            if t % query_frequency == 0:

                rng, key = jax.random.split(rng)

                curr_image = obs_to_img(last_obs, variant)
                    
                if use_local_policy:
                    # extract the feature from the pi0 VLM backbone and the kv cache
                    output = agent_dp.get_prefix_rep_and_kv_cache(obs_pi_zero)
                    kv_cache = output["kv_cache"]
                    img_rep_pi0 = output['prefix_rep'][0]
                    if save_kv_cache:
                        k, v = output["kv_cache"]
                        assert isinstance(k, jax.Array)
                        assert isinstance(v, jax.Array)
                        # k = np.asarray(k)
                        # v = np.asarray(v)
                        if put_kv_cache_on_cpu:
                            k = jax.device_put(k, jax.devices("cpu")[0])
                            v = jax.device_put(v, jax.devices("cpu")[0])
                        else:
                            if kv_cache_device is not None:
                                k = jax.device_put(k, kv_cache_device)
                                v = jax.device_put(v, kv_cache_device)
                        k_cache_outs.append(k)
                        v_cache_outs.append(v)
                else:
                    # extract the feature from the pi0 VLM backbone and concat with the qpos as states
                    output = agent_dp.get_prefix_rep(obs_pi_zero)
                    img_rep_pi0 = output['prefix_rep'][0]
                    kv_cache = None
                qpos = obs_to_qpos(last_obs, variant)
                qpos = np.concatenate([qpos, img_rep_pi0])
                if variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    }
                # Store original observation for distillation queries
                original_obs_dict = deepcopy(obs_pi_zero)
                # While agent_dp needs noise of (chunk_size, action_dim), the agent predicts noise of (action_dim,)
                # and we repeat it chunk_size times.
                if not sample_from_agent:
                    noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                    noise_repeat = jax.numpy.repeat(noise[:, -1:, :], chunk_size - noise.shape[1], axis=1)
                    noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                    actions_noise = noise[0, :agent.action_chunk_shape[0], :]
                else:
                    # sac agent predicts the noise for diffusion model
                    actions_noise = agent.sample_actions(obs_dict)
                    t == 0 and tqdm.write(colored(f'actions_noise shape: {actions_noise.shape}', "blue"))
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = np.repeat(actions_noise[-1:, :], chunk_size - actions_noise.shape[0], axis=0)
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
                action_list.append(actions_noise)
                obs_list.append(obs_dict)
                original_obs_list.append(original_obs_dict)
                agent_dp_output = agent_dp.infer(obs_pi_zero, noise=np.asarray(noise), kv_cache=kv_cache)
                action = agent_dp_output["actions"]
                executed_action_list.append(action[:query_frequency])

            action_t = action[t % query_frequency]

            if 'libero' in variant.env:
                last_obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env:
                last_obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
            obs_pi_zero = obs_to_pi_zero_input(last_obs, variant)
            curr_image = obs_to_img(last_obs, variant)
            if done:
                break
            
        tqdm.write("Trial finished. Mark as (1) Success or (0) Failure:")
        
        rewards.append(reward)
        image_list.append(curr_image)
        if use_local_policy:
            # extract the feature from the pi0 VLM backbone and the kv cache

            output = agent_dp.get_prefix_rep_and_kv_cache(obs_pi_zero)
            img_rep_pi0 = output['prefix_rep'][0]
            if save_kv_cache:
                k, v = output["kv_cache"]
                assert isinstance(k, jax.Array)
                assert isinstance(v, jax.Array)
                # k = np.asarray(k)
                # v = np.asarray(v)
                if put_kv_cache_on_cpu:
                    k = jax.device_put(k, jax.devices("cpu")[0])
                    v = jax.device_put(v, jax.devices("cpu")[0])
                else:
                    if kv_cache_device is not None:
                        k = jax.device_put(k, kv_cache_device)
                        v = jax.device_put(v, kv_cache_device)
                k_cache_outs.append(k)
                v_cache_outs.append(v)
        else:
            # extract the feature from the pi0 VLM backbone and concat with the qpos as states
            output = agent_dp.get_prefix_rep(obs_pi_zero)
            img_rep_pi0 = output['prefix_rep'][0]
        qpos = obs_to_qpos(last_obs, variant)
        qpos = np.concatenate([qpos, img_rep_pi0])
        
        if variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
            }
        # Store final original observation
        curr_image = obs_to_img(last_obs, variant)
        image_list.append(curr_image)
        original_obs_dict = deepcopy(obs_pi_zero)
        obs_list.append(obs_dict)
        original_obs_list.append(original_obs_dict)
        tqdm.write(f'Rollout Done')
        
    finally:
        # Reward in environment MDP is -1 per step until success
        # Mask is 1 until done, then 0 if success else all 1
        # is_bad is for marking really bad episodes by giving large negative reward at the end and marking episode as failure.
        
        # Match rewards in noise MDP with rewards in environment MDP
        # Noise MDP operates at a lower frequency (once every query_frequency steps)
        # So the rewards for one step in the noise MDP is the discounted sum of rewards 
        # over query_frequency steps in the environment MDP
        
        # In add_online_data_to_buffer, we set the discount factor 
        # for the noise MDP step to be discount ** query_frequency to be consistent
        
        # Using the formula for the sum of a geometric series
        # per_step_reward = -1 * (1 - variant.discount ** (query_frequency)) / (1 - variant.discount)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        is_success = (reward == variant.env_max_reward)
        print(f'Rollout Done: {episode_return=}, Success: {is_success}')
        print(f'Actions length: {len(action_list)=}')

        per_step_reward = -1
        if is_success:
            query_steps = len(action_list)
            rewards = np.concatenate([per_step_reward * np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            query_steps = len(action_list)
            # if is_bad:
            #     rewards = np.concatenate([per_step_reward * np.ones(query_steps - 1), [-50]])
            #     masks = np.concatenate([np.ones(query_steps - 1), [0]])
            # else:
            rewards = per_step_reward * np.ones(query_steps)
            masks = np.ones(query_steps)
        
        if moving_avg_success is not None:
            moving_avg_success.append(int(is_success))
            moving_avg_ep_len.append(t)
            total_avg_success = np.mean(moving_avg_success)
            total_avg_ep_len = np.mean(moving_avg_ep_len)
        if wandb_logger is not None:
            wandb_logger.log({f'is_success': int(is_success)}, step=i)
            wandb_logger.log({f'ep_len': t}, step=i)
            wandb_logger.log({f'moving_avg_success': total_avg_success}, step=i)
            wandb_logger.log({f'moving_avg_ep_len': total_avg_ep_len}, step=i)
            wandb_logger.log({f'total_num_traj': traj_id}, step=i)

        video_path = os.path.join(variant.outputdir, f'video_high_{traj_id}_{is_success}.mp4')
        video = np.stack(image_list)
        videosave(video_path, video, fps=15)
       
        tqdm.write("Episide Done!")
    
    """
    Each key is a list of length T
    Shapes of the list items for each key are as follows:
    observations {
        "pixels": (1, 3, 128, 128, 3),
        "state": (1, 2055, 1)
    }
    original_observations {
        # raw env observation dict (e.g., images, state, prompt, robot_id, etc.)
    }
    actions: (1, 32)
    executed_actions: (query_freq, 7)
    rewards: int
    masks: int
    is_success: int
    env_steps: int
    k_cache_out: (18, 1, 968, 1, 256)
    v_cache_out: (18, 1, 968, 1, 256)
    """
    traj = {
        'observations': obs_list,
        'original_observations': original_obs_list,
        'actions': action_list,
        'executed_actions': executed_action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'env_steps': t + 1,
    }
    if save_kv_cache:
        traj['original_k_cache'] = k_cache_outs
        traj['original_v_cache'] = v_cache_outs
    
    return traj