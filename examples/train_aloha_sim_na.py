#! /usr/bin/env python
import copy
import os
import time
import jax
from termcolor import cprint
from jaxrl2.agents.pixel_sac.pixel_sac_learner_na import PixelSACLearnerNA
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np
import logging

import gymnasium as gym
from gym.spaces import Dict, Box

from jaxrl2.data import ReplayBufferNA
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_aloha_sim_na import trajwise_alternating_training_loop_na
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi.training import config as openpi_config
from openpi.shared import download

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )

class DummyEnvNA(gym.ObservationWrapper):

    def __init__(self, variant, original_action_dim=14):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)

        self.num_cameras = variant.num_cameras
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8
            elif variant.env == 'aloha_cube':
                state_dim = 14
            state_dim = state_dim + 2048 # 2048 is the image representation's dim
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)

        self.observation_space = Dict(obs_dict)
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32) # 32 is the noise action space of pi 0
        self.executed_action_dim = variant.query_freq * original_action_dim
        self.original_action_space = Box(low=-1, high=1, shape=(self.executed_action_dim,), dtype=np.float32)
        cprint(f'Noise action dim: {np.prod(self.action_space.shape)}', "cyan")
        cprint(f'Executed action dim: {self.executed_action_dim}', "cyan")

def main(variant):
    if os.environ.get('dsrl_device_idx') is not None:
        all_devices = jax.devices()
        device_idx = int(os.environ['dsrl_device_idx'])
        devices = [all_devices[device_idx]]
    else:
        devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    cprint(f'num devices: {num_devices}', "green")
    cprint(f'batch size: {variant.batch_size}', "green")
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
        
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
    
    exp_folder = os.path.expanduser(os.environ['EXP'])
    exp_folder = os.path.realpath(exp_folder)
    outputdir = os.path.join(exp_folder, expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    time.sleep(3) 

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    
    if variant['use_local_policy']:
        if os.environ.get('local_policy_device') is not None:
            local_policy_device = os.environ['local_policy_device'].split(',')
            local_policy_device = [int(x) for x in local_policy_device]
            selected_devices = [all_devices[i] for i in local_policy_device]
            local_policy_mesh = jax.sharding.Mesh(selected_devices, ("x",))
            local_policy_sharding = jax.sharding.NamedSharding(local_policy_mesh, jax.sharding.PartitionSpec())
        else:
            local_policy_sharding = None
        if variant.env == 'libero':
            config = openpi_config.get_config("pi0_libero")
            checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
        elif variant.env == 'aloha_cube':
            config = openpi_config.get_config("pi0_aloha_sim")
            checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_aloha_sim")
        else:
            raise NotImplementedError()

        agent_dp = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            sharding=local_policy_sharding
        )
        cprint(f"Policy metadata: {agent_dp.metadata}", "green")
        cprint(f"Policy config: {config}", "green")
        cprint(f"Policy checkpoint dir: {checkpoint_dir}", "green")
        agent_dp._sample_kwargs["num_steps"] = variant.flow_integration_steps
    else:
        agent_dp = _websocket_client_policy.WebsocketClientPolicy(
            host=os.environ['remote_host'],
            port=os.environ['remote_port']
        )
        cprint(f"Server metadata: {agent_dp.get_server_metadata()}", "green")
    agent_dp.seed(variant.seed)

    cprint("initializing environment...", "green")
    
    from gymnasium.envs.registration import register
    register(
        id="gym_aloha/AlohaTransferCube-v0",
        entry_point="gym_aloha.env:AlohaEnv",
        max_episode_steps=400,
        nondeterministic=True,
        kwargs={"obs_type": "pixels", "task": "transfer_cube"},
    )
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    eval_env = copy.deepcopy(env)
    variant.env_max_reward = 4

    cprint("created the env!", "green")

    original_action_dim = 14
    dummy_env = DummyEnvNA(variant, original_action_dim=original_action_dim)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    sample_env_action = add_batch_dim(dummy_env.original_action_space.sample())
    cprint(f'sample obs shapes: {[(k, v.shape) for k, v in sample_obs.items()]}', "green")
    cprint(f'sample action shape: {sample_action.shape}', "green")
    cprint(f'sample env action shape: {sample_env_action.shape}', "green")
    time.sleep(3)

    robot_config = dict(
        camera_to_use='images',
        action_chunk_size=variant.action_chunk_size,
        max_timesteps=variant.max_timesteps,
        use_local_policy=variant['use_local_policy'],
        save_kv_cache=variant['save_kv_cache'],
    )
    cprint(f'robot config: {robot_config}', "green")
    time.sleep(3)
    
    agent = PixelSACLearnerNA(variant.seed, sample_obs, sample_action, env_actions=sample_env_action, noise_scale_inside=variant.noise_scale_inside, **kwargs)
    
    # Move all agent state to the DSRL device
    if len(devices) == 1 and len(jax.devices()) > 1:
        dsrl_dev = devices[0]
        agent._rng = jax.device_put(agent._rng, dsrl_dev)
        agent._actor = jax.device_put(agent._actor, dsrl_dev)
        agent._critic = jax.device_put(agent._critic, dsrl_dev)
        agent._temp = jax.device_put(agent._temp, dsrl_dev)
        agent._action_critic = jax.device_put(agent._action_critic, dsrl_dev)
        agent._target_action_critic_params = jax.device_put(agent._target_action_critic_params, dsrl_dev)
        agent.noise_scale = jax.device_put(agent.noise_scale, dsrl_dev)
    if variant.restore_path != '':
        cprint(f'restoring from {variant.restore_path}', "red")
        agent.restore_checkpoint(variant.restore_path)

    # online_buffer_size = 2 * variant.max_steps // variant.multi_grad_step
    online_buffer_size = variant.online_buffer_size
    # Get original image shape from env observation
    online_replay_buffer = ReplayBufferNA(dummy_env.observation_space, dummy_env.action_space, dummy_env.executed_action_dim, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    trajwise_alternating_training_loop_na(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp, robot_config=robot_config)
