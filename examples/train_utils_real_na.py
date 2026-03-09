"""Training utilities for DSRL-NA (Real Robot)."""
from copy import deepcopy
from jaxrl2.agents.pixel_sac import PixelSACLearnerNA
from collections import deque
import os
import time
import collections
from tqdm import tqdm
import numpy as np
import jax
from jaxrl2.utils.nonblocking_listener import keyboard_listener
from openpi_client import image_tools
from videoio import videosave
from flax.core import frozen_dict
from termcolor import colored, cprint

from examples.train_utils_na import (
    add_online_data_to_buffer_na,
    generate_distillation_batch,
    get_next_actions_from_dp,
    remove_original_obs_keys,
)

break_flag = False

def on_press(key):
    global break_flag
    tqdm.write(f"'{key}' pressed")
    if key == 'b':
        break_flag = True


def process_images(image_keys, resize_image, obs):
    '''
    concat the images from all cameras
    '''
    imgs = [image_tools.resize_with_pad(obs[k], resize_image, resize_image) for k in image_keys]
    img_all = np.concatenate(imgs, axis=2)[np.newaxis, ..., np.newaxis]
    return img_all


def trajwise_alternating_training_loop_na(variant, agent: PixelSACLearnerNA, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       shard_fn=None, agent_dp=None, robot_config=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    # if shard_fn is not None:
    #     replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)
    keys_to_remove = ['original_observations', 'original_next_observations']
    cprint(f"Distillation noise sampling strategy: {variant['grl_noise_sample']=}", "green" if variant['grl_noise_sample'] else "red")
    i = 0
    total_env_steps = 0
    total_num_traj = 0
    moving_avg_success = collections.defaultdict(lambda: deque(maxlen=20))
    moving_avg_ep_len = collections.defaultdict(lambda: deque(maxlen=20))
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
    action_critic_updates = 0
    noise_critic_updates = 0
    noise_actor_updates = 0
    with tqdm(total=variant.max_steps, initial=0, dynamic_ncols=True) as pbar:
        while i <= variant.max_steps:
            sample_from_agent = i != 0 or variant['restore_path'] != ''
            traj = collect_traj_na(variant, agent, env, i, sample_from_agent, agent_dp, wandb_logger, total_num_traj, robot_config, moving_avg_success, moving_avg_ep_len)
            total_num_traj += 1
            add_online_data_to_buffer_na(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            tqdm.write(f'online buffer timesteps length: {len(online_replay_buffer)}')
            tqdm.write(f'online buffer num traj: {total_num_traj}')
            tqdm.write(f'total env steps: {total_env_steps}')
            
            action_critic_steps = 20
            noise_critic_steps = 10
            noise_actor_steps = 1
            total_inter_steps = (action_critic_steps + noise_critic_steps + noise_actor_steps)
            if i == 0:
                num_gradsteps = len(online_replay_buffer) * variant.multi_grad_step
                num_gradsteps = len(online_replay_buffer) * total_inter_steps
                num_gradsteps = 3000  # warmup with fixed number of grad steps
            else:
                num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step
                num_gradsteps = len(traj["rewards"]) * total_inter_steps
            tqdm.write(f'num_gradsteps: {num_gradsteps}')
            distill_freq = variant['distill_freq']
            times = collections.defaultdict(list)
            if total_num_traj >= variant.num_initial_traj_collect:
                for grad_idx in tqdm(range(num_gradsteps), dynamic_ncols=True, leave=False):
                    inter_step = grad_idx % total_inter_steps
                    train_action_critic = (inter_step < action_critic_steps)
                    train_noise_critic = (inter_step >= action_critic_steps) and (inter_step < action_critic_steps + noise_critic_steps)
                    train_noise_actor = (inter_step >= action_critic_steps + noise_critic_steps)
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
                    # Generate distillation batch only every distill_freq steps
                    
                    if train_noise_critic:
                        # Fetch cache for current obs
                        # if 'indices' in batch:
                        indices = batch['indices']
                        original_k_cache = replay_buffer.get_cache(indices, 'k')
                        original_v_cache = replay_buffer.get_cache(indices, 'v')
                        batch['original_k_cache'] = original_k_cache
                        batch['original_v_cache'] = original_v_cache
                        
                        start = time.time()
                        distill_noise, distill_actions, end_times = generate_distillation_batch(batch, agent, agent_dp, robot_config, variant)
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
                        batch['original_next_k_cache'] = original_next_k_cache
                        batch['original_next_v_cache'] = original_next_v_cache

                        agent_dp_output_batched, end_times = get_next_actions_from_dp(agent_dp, batch, next_actions_noise, agent.action_chunk_shape, robot_config, variant)
                        times['next_actions_time'].append(time.time() - start)
                        [times[f'next_actions_{k}'].append(v) for k, v in end_times.items()]

                        batch_for_update = dict(batch_for_update)
                        batch_for_update["next_executed_actions"] = agent_dp_output_batched
                        batch_for_update["next_log_probs"] = next_actions_noise_log_probs
                        batch_for_update = frozen_dict.freeze(batch_for_update)
                    
                    start = time.time()
                    update_info = agent.update(batch_for_update, distill_batch=distill_batch, train_action_critic=train_action_critic, train_noise_actor=train_noise_actor)
                    times['update_time'].append(time.time() - start)

                    pbar.update()
                    i += 1
                    action_critic_updates += int(train_action_critic)
                    noise_critic_updates += int(train_noise_critic)
                    noise_actor_updates += int(train_noise_actor)
                    
                    if i % variant.log_interval == 0 or (i == 1 or i == variant.max_steps):
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
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
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1:
                        if i % variant.checkpoint_interval == 0 or (i == 1 or i == variant.max_steps):
                            agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)


def collect_traj_na(variant, agent, env, i, sample_from_agent, agent_dp=None, wandb_logger=None, traj_id=None, robot_config=None, moving_avg_success=None, moving_avg_ep_len=None):
    query_frequency = variant.query_freq
    image_keys = robot_config['image_keys']
    max_timesteps = robot_config['max_timesteps']
    chunk_size = robot_config['action_chunk_size']
    use_local_policy = robot_config["use_local_policy"]
    save_kv_cache = robot_config["save_kv_cache"]
    if save_kv_cache:
        assert use_local_policy, "save_kv_cache is true, but use_local_policy is false"
    agent._rng, rng = jax.random.split(agent._rng)
    x_min, x_max = -0.15, 0.15
    y_min, y_max = 0.2, 0.43
    z_min, z_max = 1.1, 1.25
    point = np.random.uniform(
        low=[x_min, y_min, z_min], high=[x_max, y_max, z_max]
    )
    if variant['rand_start']:
        last_obs, last_info = env.reset(rand_point=point)
    else:
        last_obs, last_info = env.reset()
    if "test_prompt" in variant and variant['test_prompt'] is not None:
        last_obs['prompt'] = variant['test_prompt']
        tqdm.write(colored(f'Setting test prompt: {variant["test_prompt"]}', 'yellow'))
    else:
        tqdm.write(colored(f'Using environment prompt: {last_obs["prompt"]}', 'yellow'))
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
    success_queue = deque(maxlen=10)

    try:
        global break_flag
        break_flag = False
        disable = False
        with keyboard_listener(on_press=on_press, disable=disable) as keyboard:
            for t in tqdm(range(max_timesteps), dynamic_ncols=True):        
                if break_flag:
                    print("'b' pressed, stopping rollout.")
                    break        
                image_list.append(last_obs[robot_config['camera_to_use']])

                if t % query_frequency == 0:

                    rng, key = jax.random.split(rng)

                    img_all = process_images(image_keys, variant.resize_image, last_obs)
                    
                    if use_local_policy:
                        # extract the feature from the pi0 VLM backbone and the kv cache
                        output = agent_dp.get_prefix_rep_and_kv_cache(last_obs)
                        kv_cache = output["kv_cache"]
                        img_rep_pi0 = output['prefix_rep'][0]
                        if save_kv_cache:
                            k, v = output["kv_cache"]
                            assert isinstance(k, jax.Array)
                            assert isinstance(v, jax.Array)
                            k_cache_outs.append(k)
                            v_cache_outs.append(v)
                    else:
                        # extract the feature from the pi0 VLM backbone and concat with the qpos as states
                        output = agent_dp.get_prefix_rep(last_obs)
                        img_rep_pi0 = output['prefix_rep'][0]
                        kv_cache = None
                    qpos = np.concatenate([last_obs["observation/state"], img_rep_pi0])

                    obs_dict = {
                        'pixels': img_all,
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                    # Store original observation for distillation queries
                    original_obs_dict = deepcopy(last_obs)
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
                    agent_dp_output = agent_dp.infer(last_obs, noise=np.asarray(noise), kv_cache=kv_cache)
                    action = agent_dp_output["actions"]
                    executed_action_list.append(action[:query_frequency])

                action_t = action[t % query_frequency]
                # binarize gripper action.
                if action_t[-1].item() > 0.5:
                    action_t = np.concatenate([action_t[:-1], np.ones((1,))])
                else:
                    action_t = np.concatenate([action_t[:-1], np.zeros((1,))])
                if "progress" in agent_dp_output:
                    success_queue.append(agent_dp_output['progress'][t % query_frequency])

                action_t = np.concatenate([action_t, [last_obs['robot_id']]])
                assert action_t.shape == (8,)
                last_obs, reward, term, trunc, last_info = env.step(action_t)
                if "test_prompt" in variant and variant['test_prompt'] is not None:
                    last_obs['prompt'] = variant['test_prompt']
                done = term or trunc or (len(success_queue) == success_queue.maxlen and np.mean(success_queue) >= variant['success_threshold'])
                if done:
                    break
                
        print("Trial finished. Mark as (1) Success or (0) Failure:")
        is_success = False
        is_bad = False
        valid_opts = ['c', 'r', 'y', 'yes', 'n', 'no', 'x']
        save = 'esc'
        rollout_update = 1
        while save not in valid_opts:
            save = input("Trial finished. Mark as (y) Success or (n) Failure: (press c to exit)")
            save = save.lower()
            time.sleep(0.01)
        if save == 'c':
            print("Exiting")
            exit(0)
        elif save == 'r':
            print("Retrying episode")
            raise Exception("Retry Episode")
        elif save in ['y', 'yes']:
            tqdm.write(colored("Episode marked as SUCCESS.", "green"))
            is_success = True
        elif save in ['n', 'no']:
            tqdm.write(colored("Episode marked as FAILURE.", "red"))                   
            is_success = False
        elif save == 'x':
            tqdm.write(colored("Episode marked as BAD. Will be given -50 as last step reward.", "red"))                   
            is_success = False
            is_bad = True

        image_list.append(last_obs[robot_config['camera_to_use']])
        img_all = process_images(image_keys, variant.resize_image, last_obs)
        if use_local_policy:
            # extract the feature from the pi0 VLM backbone and the kv cache
            output = agent_dp.get_prefix_rep_and_kv_cache(last_obs)
            img_rep_pi0 = output['prefix_rep'][0]
            if save_kv_cache:
                k, v = output["kv_cache"]
                assert isinstance(k, jax.Array)
                assert isinstance(v, jax.Array)
                k_cache_outs.append(k)
                v_cache_outs.append(v)
        else:
            # extract the feature from the pi0 VLM backbone and concat with the qpos as states
            output = agent_dp.get_prefix_rep(last_obs)
            img_rep_pi0 = output['prefix_rep'][0]
        qpos = np.concatenate([last_obs["observation/state"], img_rep_pi0])
        
        obs_dict = {
            'pixels': img_all,
            'state': qpos[np.newaxis, ..., np.newaxis],
        }
        # Store final original observation
        original_obs_dict = deepcopy(last_obs)
        obs_list.append(obs_dict)
        original_obs_list.append(original_obs_dict)
        tqdm.write(f'Rollout Done')
        keyboard.stop_listening()
        
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
        per_step_reward = -1 * (1 - variant.discount ** (query_frequency)) / (1 - variant.discount)
        if is_success:
            query_steps = len(action_list)
            rewards = np.concatenate([per_step_reward * np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            query_steps = len(action_list)
            if is_bad:
                rewards = np.concatenate([per_step_reward * np.ones(query_steps - 1), [-50]])
                masks = np.concatenate([np.ones(query_steps - 1), [0]])
            else:
                rewards = per_step_reward * np.ones(query_steps)
                masks = np.ones(query_steps)
        
        if moving_avg_success is not None:
            moving_avg_success["real"].append(int(is_success))
            moving_avg_ep_len["real"].append(t)
            total_avg_success = np.mean([np.mean(v) for v in moving_avg_success.values()])
            total_avg_ep_len = np.mean([np.mean(v) for v in moving_avg_ep_len.values()])
            task_avg_success = np.mean(moving_avg_success["real"])
            task_avg_ep_len = np.mean(moving_avg_ep_len["real"])
        if wandb_logger is not None:
            wandb_logger.log({f'is_success': int(is_success)}, step=i)
            wandb_logger.log({f'ep_len': t}, step=i)
            wandb_logger.log({f'moving_avg_success': total_avg_success}, step=i)
            wandb_logger.log({f'moving_avg_ep_len': total_avg_ep_len}, step=i)
            wandb_logger.log({f'total_num_traj': traj_id}, step=i)

        video_path = os.path.join(variant.outputdir, f'video_high_{traj_id}_{is_success}.mp4')
        video = np.stack(image_list)
        videosave(video_path, video, fps=15)
       
        tqdm.write("Episide Done! Press c after resetting the environment")
    
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
